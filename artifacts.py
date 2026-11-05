"""Markdown artifact workspace: the only place agent actions can cause durable writes.

Agents never supply paths. They name an artifact ("world", "world.md" or "artifacts/world.md"); the name is
validated against a strict pattern and mapped to <workspace>/<name>.md. Symlinks, traversal, non-Markdown
files, oversized content and stale overwrites are rejected. Content is stored as data and never interpreted.
"""
import hashlib
import os
import re
import tempfile
from pathlib import Path

NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,47}$")
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class ArtifactError(Exception):
    """An artifact operation was refused."""


class ArtifactManager:
    def __init__(self, root, max_count=10, max_bytes=12000, max_total_bytes=80000):
        root = Path(root)
        if root.is_symlink():
            raise ArtifactError("workspace root must not be a symlink")
        root.mkdir(parents=True, exist_ok=True)
        self.root = root.resolve()
        self.max_count, self.max_bytes, self.max_total = max_count, max_bytes, max_total_bytes
        self.versions = {}   # file name -> version (only artifacts created through this manager exist)
        self.seen = {}       # (agent, file name) -> last version that agent read or wrote
        self.log = []        # dicts: turn, agent, op, name, version, ok, detail

    # ── validation ──
    def normalize(self, name):
        if not isinstance(name, str):
            raise ArtifactError("artifact name must be text")
        n = name.strip().lower()
        if n.startswith("artifacts/"):
            n = n[len("artifacts/"):]
        if n.endswith(".md"):
            n = n[:-3]
        if not NAME_RE.match(n):
            raise ArtifactError(f"invalid artifact name {name[:60]!r}: use letters, digits, '-' or '_' (no paths)")
        return n + ".md"

    def _path(self, fname):
        path = self.root / fname
        if path.is_symlink():
            raise ArtifactError("symlinked artifacts are not allowed")
        if os.path.dirname(os.path.realpath(path)) != str(self.root):
            raise ArtifactError("artifact resolves outside the workspace")
        if path.exists() and not path.is_file():
            raise ArtifactError("artifact is not a regular file")
        return path

    def _total_bytes(self, exclude=None):
        total = 0
        for f in self.versions:
            if f != exclude:
                try:
                    total += os.lstat(self.root / f).st_size
                except OSError:
                    pass
        return total

    # ── file IO ──
    def _atomic_write(self, path, text):
        fd, tmp = tempfile.mkstemp(dir=self.root, prefix=".tmp-", suffix=".part")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(text.encode("utf-8"))
                f.flush()
                os.fsync(f.fileno())
            os.chmod(tmp, 0o644)   # never executable
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def _read_file(self, path):
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(fd, "rb") as f:
            return f.read(self.max_bytes + 1).decode("utf-8", "replace")

    # ── public operations ──
    def names(self):
        return sorted(self.versions)

    def listing(self):
        return [(f, self.versions[f], os.lstat(self.root / f).st_size) for f in self.names()]

    def snapshot(self):
        files = {}
        for name in self.names():
            content = self._read_file(self._path(name))
            files[name] = dict(content=content, version=self.versions[name],
                               sha256=hashlib.sha256(content.encode("utf-8")).hexdigest())
        return dict(files=files, seen=[[agent, name, version]
                                      for (agent, name), version in self.seen.items()], log=self.log[:])

    def restore(self, snapshot):
        files = snapshot["files"]
        if len(files) > self.max_count:
            raise ArtifactError("saved workspace exceeds artifact count limit")
        total_bytes = 0
        for name, saved in files.items():
            if self.normalize(name) != name:
                raise ArtifactError("invalid saved artifact name")
            path = self._path(name)
            if path.exists() and name not in self.versions:
                raise ArtifactError("refusing to overwrite an existing untracked artifact")
            encoded = saved["content"].encode("utf-8")
            if hashlib.sha256(encoded).hexdigest() != saved["sha256"]:
                raise ArtifactError(f"saved artifact checksum mismatch: {name}")
            if not isinstance(saved["version"], int) or saved["version"] < 1:
                raise ArtifactError("invalid saved artifact version")
            if len(encoded) > self.max_bytes:
                raise ArtifactError("saved artifact exceeds size limit")
            total_bytes += len(encoded)
        if total_bytes > self.max_total:
            raise ArtifactError("saved workspace exceeds total size limit")
        seen = {(agent, name): version for agent, name, version in snapshot["seen"]}
        for (_, name), version in seen.items():
            if name not in files or not 1 <= version <= files[name]["version"]:
                raise ArtifactError("invalid saved artifact read version")
        for name, saved in files.items():
            self._atomic_write(self._path(name), saved["content"])
        self.versions = {name: saved["version"] for name, saved in files.items()}
        self.seen = seen
        self.log = list(snapshot["log"])

    def read(self, agent, name, turn=0):
        try:
            fname = self.normalize(name)
            if fname not in self.versions:
                raise ArtifactError(f"no artifact named {fname}")
            text = self._read_file(self._path(fname))
        except ArtifactError as e:
            self.log.append(dict(turn=turn, agent=agent, op="read", name=str(name)[:60], version=None, ok=False, detail=str(e)))
            raise
        self.seen[(agent, fname)] = self.versions[fname]
        self.log.append(dict(turn=turn, agent=agent, op="read", name=fname, version=self.versions[fname], ok=True, detail=""))
        return fname, self.versions[fname], text

    def write(self, agent, name, mode, content, expected_version=None, turn=0, attribution=True):
        try:
            fname, version = self._write(agent, name, mode, content, expected_version, turn, attribution)
        except ArtifactError as e:
            self.log.append(dict(turn=turn, agent=agent, op=str(mode)[:10], name=str(name)[:60], version=None, ok=False, detail=str(e)))
            raise
        self.log.append(dict(turn=turn, agent=agent, op=mode, name=fname, version=version, ok=True, detail=f"{len(content)} chars"))
        return fname, version

    def _write(self, agent, name, mode, content, expected_version, turn, attribution):
        if mode not in ("create", "append", "replace"):
            raise ArtifactError("mode must be create, append or replace")
        if not isinstance(content, str) or not content.strip():
            raise ArtifactError("artifact content must be non-empty text")
        content = CONTROL_CHARS.sub("", content)
        fname = self.normalize(name)
        path = self._path(fname)
        exists = fname in self.versions
        if not exists and len(self.versions) >= self.max_count:
            raise ArtifactError(f"artifact limit reached ({self.max_count})")
        if not exists and path.exists():
            raise ArtifactError("a file with that name exists but is not a chamber artifact")
        current = self.versions.get(fname, 0)
        if mode == "create" and exists:
            raise ArtifactError(f"{fname} already exists; use append or replace")
        if expected_version is not None:
            try:
                expected_version = int(expected_version)
            except (TypeError, ValueError):
                raise ArtifactError("expected_version must be a number")
            if expected_version != current:
                raise ArtifactError(f"{fname} is at v{current}, not v{expected_version}; read it again")
        if mode == "replace" and exists and expected_version is None and self.seen.get((agent, fname)) != current:
            raise ArtifactError(f"{fname} changed since you last saw it (now v{current}); read it before replacing")
        if mode == "append" and exists:
            old = self._read_file(path)
            note = f"\n\n— *{agent}, turn {turn}*" if attribution else ""
            new = old.rstrip("\n") + "\n\n" + content.strip() + note + "\n"
        else:
            new = content.strip() + "\n"
        size = len(new.encode("utf-8"))
        if size > self.max_bytes:
            raise ArtifactError(f"{fname} would be {size} bytes (limit {self.max_bytes})")
        if self._total_bytes(exclude=fname) + size > self.max_total:
            raise ArtifactError(f"workspace size limit reached ({self.max_total} bytes)")
        self._atomic_write(path, new)
        self.versions[fname] = current + 1
        self.seen[(agent, fname)] = current + 1
        return fname, current + 1
