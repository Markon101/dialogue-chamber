"""Python-first process management and model identity verification for llama.cpp servers.

Manages Snapdragon/Termux server processes, tracks PID/PGID ownership, verifies model identity,
prevents conflicting server clobbering, and ensures safe shutdown without broad pkill calls.
"""
import json
import fcntl
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from contextlib import contextmanager
from typing import Any, Dict, Optional, Tuple

CHAMBER_DIR = Path(__file__).resolve().parent
SNAPDRAGON_DIR = Path(os.environ.get("SNAPDRAGON_DIR", Path.home() / "projects" / "llama-snapdragon"))
DEFAULT_SERVE_SCRIPT = CHAMBER_DIR / "guarded_serve_best.sh"
DEFAULT_STATE_DIR = CHAMBER_DIR / ".locks" / "servers"

# Canonical model aliases
MODEL_ALIASES: Dict[str, Tuple[str, str, int]] = {
    "gemma": ("gemma4-htp", "gemma-4-E4B-it-Q4_0.gguf", 32768),
    "gemma4": ("gemma4-htp", "gemma-4-E4B-it-Q4_0.gguf", 32768),
    "gemma4-htp": ("gemma4-htp", "gemma-4-E4B-it-Q4_0.gguf", 32768),
    "gemma-mtp": ("gemma4-cpu-mtp", "gemma-4-E4B-it-Q4_0.gguf", 32768),
    "gemma4-mtp": ("gemma4-cpu-mtp", "gemma-4-E4B-it-Q4_0.gguf", 32768),
    "gemma4-cpu-mtp": ("gemma4-cpu-mtp", "gemma-4-E4B-it-Q4_0.gguf", 32768),
    "llama1b": ("llama1b-htp", "Llama-3.2-1B-Instruct-Q4_0.gguf", 131072),
    "llama-1b": ("llama1b-htp", "Llama-3.2-1B-Instruct-Q4_0.gguf", 131072),
    "llama1b-htp": ("llama1b-htp", "Llama-3.2-1B-Instruct-Q4_0.gguf", 131072),
    "llama1b-htp-q8": ("llama1b-htp-q8", "Llama-3.2-1B-Instruct-Q4_0.gguf", 8192),
    "gemma4-htp-q8": ("gemma4-htp-q8", "gemma-4-E4B-it-Q4_0.gguf", 8192),
    "gemma4-htp-mtp": ("gemma4-htp-mtp", "gemma-4-E4B-it-Q4_0.gguf", 4096),
}


def process_start_ticks(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def running_llama_servers():
    """List visible llama-server processes; fail closed if process inspection fails."""
    try:
        result = subprocess.run(["ps", "-ef"], capture_output=True, text=True, timeout=5, check=True)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("Cannot inspect running processes; refusing a local server operation") from exc
    servers = []
    lines = result.stdout.splitlines()
    if not lines or "PID" not in lines[0] or "CMD" not in lines[0]:
        raise RuntimeError("Cannot interpret process list; refusing a local server operation")
    for line in lines[1:]:
        fields = line.split(None, 7)
        if len(fields) != 8 or not fields[1].isdigit():
            continue
        executable = fields[7].split(None, 1)[0]
        if Path(executable).name.startswith("llama-server"):
            servers.append((int(fields[1]), executable))
    return servers


def assert_single_llama_server():
    servers = running_llama_servers()
    if len(servers) > 1:
        raise RuntimeError("Multiple llama-server processes are running: " +
                           ", ".join(str(pid) for pid, _ in servers) + "; stop the extra servers before continuing")
    return servers


@contextmanager
def server_launch_lock():
    """Serialize cooperating launchers; Dory keeps the descriptor across exec."""
    DEFAULT_STATE_DIR.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(DEFAULT_STATE_DIR / "server_launch.lock"), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another guarded local server launch is active") from exc
        os.set_inheritable(fd, True)
        yield
    finally:
        os.close(fd)


def resolve_model(alias_or_name: str) -> Tuple[str, str, int]:
    """Resolves a model alias or filename into (mode, model_filename, default_ctx)."""
    key = (alias_or_name or "gemma4-htp").strip().lower()
    if key in MODEL_ALIASES:
        return MODEL_ALIASES[key]

    # Check by filename basename
    base = Path(alias_or_name).name
    for mode, fname, ctx in MODEL_ALIASES.values():
        if base.lower() == fname.lower():
            return mode, fname, ctx

    # Custom or unknown model
    return alias_or_name, base, 4096


class LlamaServerManager:
    """Controls llama-server lifecycle with strict process ownership and model validation."""

    def __init__(
        self,
        serve_script: Optional[Path] = None,
        state_dir: Optional[Path] = None,
        timeout_s: int = 90,
    ):
        self.serve_script = serve_script or DEFAULT_SERVE_SCRIPT
        self.state_dir = state_dir or DEFAULT_STATE_DIR
        self.timeout_s = timeout_s
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def _state_file(self, port: int) -> Path:
        return self.state_dir / f"server_{port}.json"

    def check_health(self, host: str = "127.0.0.1", port: int = 8080, timeout: float = 2.0) -> bool:
        """Verifies if a server on host:port is running and responsive."""
        for path in ("/health", "/v1/models"):
            url = f"http://{host}:{port}{path}"
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "DialogueChamber/2.0"})
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    if resp.status == 200:
                        return True
            except Exception:
                continue
        return False

    def query_server_info(self, host: str = "127.0.0.1", port: int = 8080, timeout: float = 2.0) -> Dict[str, Any]:
        """Queries /v1/models, /props, and /slots to discover model identity and reported context."""
        info: Dict[str, Any] = {
            "online": False,
            "model_id": None,
            "model_name": None,
            "context_window": None,
            "allocated_context": None,
            "context_source": "unknown",
            "slots": None,
            "raw_models": None,
        }

        # 1. Query /v1/models
        url_models = f"http://{host}:{port}/v1/models"
        try:
            req = urllib.request.Request(url_models, headers={"User-Agent": "DialogueChamber/2.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                info["online"] = True
                info["raw_models"] = data
                data_list = data.get("data") or []
                if data_list and isinstance(data_list, list):
                    m0 = data_list[0]
                    mid = m0.get("id", "")
                    info["model_id"] = mid
                    info["model_name"] = Path(mid).name if mid else None
                    # Some llama.cpp builds include meta in /v1/models
                    meta = m0.get("meta") or {}
                    if meta.get("n_ctx_train"):
                        info["context_window"] = int(meta["n_ctx_train"])
                        info["context_source"] = "server-reported"
        except Exception:
            pass

        # 2. Query /props for default generation settings (n_ctx)
        url_props = f"http://{host}:{port}/props"
        try:
            req = urllib.request.Request(url_props, headers={"User-Agent": "DialogueChamber/2.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                info["online"] = True
                gen = data.get("default_generation_settings") or {}
                if gen.get("n_ctx"):
                    info["context_window"] = int(gen["n_ctx"])
                    info["allocated_context"] = int(gen["n_ctx"])
                    info["context_source"] = "server-reported"
        except Exception:
            pass

        # 3. Query /slots for slot context
        url_slots = f"http://{host}:{port}/slots"
        try:
            req = urllib.request.Request(url_slots, headers={"User-Agent": "DialogueChamber/2.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                info["online"] = True
                info["slots"] = data
                if isinstance(data, list) and len(data) > 0:
                    s0 = data[0]
                    if s0.get("n_ctx"):
                        info["context_window"] = int(s0["n_ctx"])
                        info["allocated_context"] = int(s0["n_ctx"])
                        info["context_source"] = "server-reported"
                    if not info["model_name"] and s0.get("model"):
                        info["model_name"] = Path(s0["model"]).name
        except Exception:
            pass

        return info

    def is_server_owned(self, port: int = 8080) -> bool:
        """Determines if the server on port was started by us and is still alive."""
        st_file = self._state_file(port)
        if not st_file.exists():
            return False

        try:
            data = json.loads(st_file.read_text(encoding="utf-8"))
            pid = int(data.get("pid", 0))
            if pid <= 0:
                return False

            os.kill(pid, 0)
            if not data.get("start_ticks") or data["start_ticks"] != process_start_ticks(pid):
                return False

            # Validate that /proc/<pid>/cmdline matches our expected processes
            cmdline_path = Path(f"/proc/{pid}/cmdline")
            if cmdline_path.exists():
                cmd = cmdline_path.read_text(encoding="utf-8", errors="ignore").replace("\0", " ")
                if any(Path(part).name in ("llama-server", "guard.sh", "guarded_serve_best.sh") for part in cmd.split()):
                    return True
            else:
                return False
        except (OSError, ValueError):
            pass

        # Stale state file
        st_file.unlink(missing_ok=True)
        return False

    def get_owned_pid(self, port: int = 8080) -> Optional[int]:
        if self.is_server_owned(port):
            try:
                data = json.loads(self._state_file(port).read_text(encoding="utf-8"))
                return int(data.get("pid", 0))
            except Exception:
                return None
        return None

    def ensure_server(
        self,
        host: str = "127.0.0.1",
        port: int = 8080,
        model_mode: str = "gemma4-htp",
        model_file: str = "gemma-4-E4B-it-Q4_0.gguf",
        log_file: Optional[Path] = None,
        allow_switch: bool = False,
        ctx_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Serialize launches and refuse to create a second llama-server."""
        with server_launch_lock():
            return self._ensure_server(host, port, model_mode, model_file, log_file, allow_switch, ctx_size)

    def _ensure_server(self, host, port, model_mode, model_file, log_file, allow_switch, ctx_size):
        """Ensures the requested model is running on host:port, refusing mismatches unless switch authorized."""
        log_path = log_file or (CHAMBER_DIR / "runs" / "local_server.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)

        req_base = Path(model_file).name

        servers = assert_single_llama_server()
        healthy = self.check_health(host, port, timeout=2.0)
        if servers and not healthy:
            raise RuntimeError(f"A llama-server (PID {servers[0][0]}) is already running elsewhere or is not ready; refusing to start another")
        if healthy:
            info = self.query_server_info(host, port, timeout=2.0)
            loaded = info.get("model_name") or info.get("model_id")
            if not loaded:
                raise RuntimeError("Cannot verify the running server's model identity")
            if self.is_server_owned(port):
                saved = json.loads(self._state_file(port).read_text())
                if saved.get("mode") != model_mode:
                    raise RuntimeError("Running server mode differs; stop the owned server before changing backend or MTP mode")
            elif "mtp" in model_mode:
                raise RuntimeError("Cannot verify ownership and mode of the requested MTP server")
            if "mtp" in model_mode and not any(slot.get("speculative") for slot in (info.get("slots") or [])):
                raise RuntimeError("Requested MTP server does not report speculative decoding")

            if loaded and Path(loaded).name != req_base:
                if allow_switch:
                    if self.is_server_owned(port):
                        print(f"[ServerManager] Switching model from {loaded} to {req_base} on port {port}...")
                        self.stop_owned_server(port)
                        time.sleep(1.5)
                    else:
                        raise RuntimeError(
                            f"Port {port} is occupied by an UNOWNED/EXTERNAL server running '{loaded}'. "
                            f"Cannot switch to '{req_base}' because external servers must not be killed. "
                            f"Use a different port or stop it manually."
                        )
                else:
                    raise RuntimeError(
                        f"Model mismatch on http://{host}:{port}! "
                        f"Currently running: '{loaded}', but requested: '{req_base}' ({model_mode}). "
                        f"Pass --allow-server-switch to restart an owned server, or specify matching model."
                    )
            else:
                return {
                    "action": "reused",
                    "host": host,
                    "port": port,
                    "model_name": loaded or req_base,
                    "info": info,
                }

        # Server not running or safely stopped; launch it
        if not self.serve_script.exists():
            raise FileNotFoundError(f"Server launch script not found: {self.serve_script}")

        env = dict(os.environ)
        if ctx_size:
            env["CTX_SIZE"] = str(ctx_size)

        log_f = open(log_path, "a", encoding="utf-8")
        try:
            # Launch in a new process group with shell=False
            proc = subprocess.Popen(
                ["bash", str(self.serve_script), model_mode, str(port)],
                env=env,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                shell=False,
                preexec_fn=os.setsid if hasattr(os, "setsid") else None,
            )
        finally:
            log_f.close()

        # Save ownership state
        st_file = self._state_file(port)
        st_file.write_text(
            json.dumps(
                {
                    "pid": proc.pid,
                    "start_ticks": process_start_ticks(proc.pid),
                    "pgid": proc.pid,
                    "port": port,
                    "host": host,
                    "mode": model_mode,
                    "model": req_base,
                    "started_at": int(time.time()),
                    "log_file": str(log_path),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        # Wait for readiness
        t0 = time.monotonic()
        ready = False
        while (time.monotonic() - t0) < self.timeout_s:
            if self.check_health(host, port, timeout=1.0):
                ready = True
                break

            if proc.poll() is not None:
                st_file.unlink(missing_ok=True)
                tail_lines = ""
                if log_path.exists():
                    try:
                        lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
                        tail_lines = "\n".join(lines[-25:])
                    except Exception:
                        pass
                raise RuntimeError(
                    f"llama-server process {proc.pid} exited with code {proc.returncode} during startup.\n"
                    f"Recent log output:\n{tail_lines}"
                )
            time.sleep(1.0)

        if not ready:
            self.stop_owned_server(port)
            tail_lines = ""
            if log_path.exists():
                try:
                    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
                    tail_lines = "\n".join(lines[-25:])
                except Exception:
                    pass
            raise TimeoutError(
                f"llama-server on http://{host}:{port} failed to become healthy within {self.timeout_s}s.\n"
                f"Recent log output:\n{tail_lines}"
            )

        info = self.query_server_info(host, port, timeout=2.0)
        return {
            "action": "started",
            "host": host,
            "port": port,
            "pid": proc.pid,
            "model_name": req_base,
            "info": info,
        }

    def stop_owned_server(self, port: int = 8080) -> bool:
        """Safely terminates an owned server on port. Returns False if unowned or already stopped."""
        if not self.is_server_owned(port):
            return False
        st_file = self._state_file(port)
        if not st_file.exists():
            return False

        try:
            data = json.loads(st_file.read_text(encoding="utf-8"))
            pid = int(data.get("pid", 0))
        except Exception:
            st_file.unlink(missing_ok=True)
            return False

        if pid <= 0:
            st_file.unlink(missing_ok=True)
            return False

        try:
            # Check if alive
            os.kill(pid, 0)
        except OSError:
            st_file.unlink(missing_ok=True)
            return False

        # Terminate process group safely
        try:
            if hasattr(os, "killpg"):
                os.killpg(pid, signal.SIGTERM)
            else:
                os.kill(pid, signal.SIGTERM)
        except OSError:
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass

        # Wait up to 10s for termination
        for _ in range(20):
            try:
                os.kill(pid, 0)
                time.sleep(0.5)
            except OSError:
                break
        else:
            # Force kill if still lingering
            if not self.is_server_owned(port):
                return False
            try:
                if hasattr(os, "killpg"):
                    os.killpg(pid, signal.SIGKILL)
                else:
                    os.kill(pid, signal.SIGKILL)
            except OSError:
                pass

        st_file.unlink(missing_ok=True)
        return True
