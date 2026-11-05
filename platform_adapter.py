"""Platform abstraction layer for Android Termux, Linux, macOS, and Windows.

Provides safe wake-lock management, file-based concurrency locking, process signals,
and system diagnostics (memory, thermal, battery) without hard dependencies.
"""
import atexit
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Platform flags
IS_TERMUX = "TERMUX_VERSION" in os.environ or "/data/data/com.termux" in sys.executable
IS_LINUX = sys.platform.startswith("linux")
IS_MACOS = sys.platform == "darwin"
IS_WINDOWS = sys.platform == "win32"

try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False

try:
    import msvcrt
    HAS_MSVCRT = True
except ImportError:
    HAS_MSVCRT = False


class WakeLockManager:
    """Termux Android wake-lock manager with multi-process reference counting."""

    def __init__(self, lock_dir: Optional[Path] = None):
        self.lock_dir = lock_dir or (Path(__file__).resolve().parent / ".locks" / "wakelock_holders")
        self.held = False
        self._registered = False

    def acquire(self) -> bool:
        if not IS_TERMUX:
            return False

        tool = shutil.which("termux-wake-lock")
        if not tool:
            return False

        try:
            self.lock_dir.mkdir(parents=True, exist_ok=True)
            holder = self.lock_dir / str(os.getpid())
            holder.touch()
            subprocess.run([tool], capture_output=True, timeout=5)
            self.held = True
            if not self._registered:
                atexit.register(self.release)
                self._registered = True
            return True
        except Exception:
            return False

    def release(self) -> bool:
        if not self.held:
            return False

        self.held = False
        try:
            holder = self.lock_dir / str(os.getpid())
            if holder.exists():
                holder.unlink()

            # Clean up dead holders
            active = 0
            if self.lock_dir.exists():
                for p in self.lock_dir.iterdir():
                    try:
                        pid = int(p.name)
                        os.kill(pid, 0)
                        active += 1
                    except (ValueError, OSError):
                        p.unlink(missing_ok=True)

            if active == 0:
                tool = shutil.which("termux-wake-unlock")
                if tool:
                    subprocess.run([tool], capture_output=True, timeout=5)
            return True
        except Exception:
            return False


class ConcurrencyLock:
    """Inter-process experiment mutex using fcntl.flock (POSIX) or file locking."""

    def __init__(self, lock_file: Optional[Path] = None):
        self.lock_file = lock_file or (Path(__file__).resolve().parent / ".locks" / "chamber_run.lock")
        self.pid_file = self.lock_file.with_suffix(".pid")
        self._fd: Optional[int] = None
        self._acquired = False

    def acquire(self, timeout_s: float = 0.0) -> bool:
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._fd = os.open(str(self.lock_file), os.O_CREAT | os.O_RDWR)
        except OSError as e:
            raise RuntimeError(f"Failed to open lock file {self.lock_file}: {e}")

        if HAS_FCNTL:
            start_t = os.times().elapsed
            while True:
                try:
                    fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    self._acquired = True
                    break
                except (BlockingIOError, OSError):
                    if timeout_s <= 0 or (os.times().elapsed - start_t) >= timeout_s:
                        holder = self.get_holder_pid() or "unknown"
                        os.close(self._fd)
                        self._fd = None
                        raise RuntimeError(
                            f"Another Dialogue Chamber experiment is running (PID: {holder}). "
                            "Wait for completion or use --force-unlock."
                        )
                    import time
                    time.sleep(0.5)

        elif HAS_MSVCRT:
            try:
                msvcrt.locking(self._fd, msvcrt.LK_NBLCK, 1)
                self._acquired = True
            except (OSError, IOError):
                holder = self.get_holder_pid() or "unknown"
                os.close(self._fd)
                self._fd = None
                raise RuntimeError(f"Lock held by PID {holder} on Windows.")
        else:
            # Fallback when no platform file locking is available
            self._acquired = True

        try:
            self.pid_file.write_text(str(os.getpid()), encoding="utf-8")
        except OSError:
            pass

        atexit.register(self.release)
        return True

    def release(self) -> None:
        if not self._acquired or self._fd is None:
            return

        try:
            if self.pid_file.exists():
                txt = self.pid_file.read_text(encoding="utf-8").strip()
                if txt == str(os.getpid()):
                    self.pid_file.unlink(missing_ok=True)
        except Exception:
            pass

        try:
            if HAS_FCNTL:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            elif HAS_MSVCRT:
                msvcrt.locking(self._fd, msvcrt.LK_UNLCK, 1)
            os.close(self._fd)
        except Exception:
            pass
        finally:
            self._fd = None
            self._acquired = False

    def get_holder_pid(self) -> Optional[str]:
        if self.pid_file.exists():
            try:
                return self.pid_file.read_text(encoding="utf-8").strip()
            except Exception:
                return None
        return None

    def force_unlock(self) -> None:
        if self.pid_file.exists():
            self.pid_file.unlink(missing_ok=True)
        if self.lock_file.exists():
            self.lock_file.unlink(missing_ok=True)


def get_available_memory_mb() -> Optional[int]:
    """Returns MemAvailable in MB from /proc/meminfo if on Linux/Termux, or estimated."""
    if os.path.exists("/proc/meminfo"):
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        parts = line.split()
                        return int(parts[1]) // 1024
        except Exception:
            pass
    return None


def get_thermal_diagnostics() -> Dict[str, Any]:
    """Gathers thermal / battery status on Termux / Linux when available."""
    diag: Dict[str, Any] = {"available": False}
    # Check termux-battery-status
    if IS_TERMUX:
        tool = shutil.which("termux-battery-status")
        if tool:
            try:
                res = subprocess.run([tool], capture_output=True, text=True, timeout=3)
                if res.returncode == 0:
                    data = json.loads(res.stdout)
                    diag["battery"] = data
                    diag["available"] = True
            except Exception:
                pass

    # Check /sys/class/thermal
    thermal_dir = Path("/sys/class/thermal")
    if thermal_dir.exists():
        temps = {}
        try:
            for zone in thermal_dir.glob("thermal_zone*"):
                temp_file = zone / "temp"
                type_file = zone / "type"
                if temp_file.exists() and os.access(temp_file, os.R_OK):
                    try:
                        raw_temp = int(temp_file.read_text().strip())
                        ztype = type_file.read_text().strip() if type_file.exists() else zone.name
                        temps[ztype] = round(raw_temp / 1000.0, 1) if raw_temp > 1000 else raw_temp
                    except Exception:
                        continue
            if temps:
                diag["thermal_zones"] = temps
                diag["available"] = True
        except Exception:
            pass

    return diag
