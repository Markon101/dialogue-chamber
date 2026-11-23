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

_NUM_SECTION_PREFIX = (
    r"(?:(?:Section|Part|Chapter)[ \t]+)?"
    r"(?:\d+(?:\.\d+)*|[IVX]{1,5})"
    r"[ \t]*[:.)\-][ \t]+"
)


def section_heading_regex(section_spec: str) -> re.Pattern:
    """Compile a tolerant, line-anchored regex matching a Markdown or bracketed section heading.

    Matches equivalent forms such as ``[Intro]``, ``## [Intro]``, ``## Intro``,
    ``## 1. Intro``, ``## [1. Intro]``, and hyphen/dash variations without
    matching ordinary body prose sentences.
    """
    core = str(section_spec or "").strip().strip("[]").strip()
    parts = re.split(r"(\s+|[-–—])", core)
    core_pat = "".join(
        r"[ \t]+" if p.isspace()
        else (r"[-–—]" if p in ("-", "–", "—") else re.escape(p))
        for p in parts if p
    )
    if not core_pat:
        return re.compile(r"(?!)")
    trail = r"(?:[ \t]*[:\-–—].*|[ \t]+\([^)]*\))?"
    b_hashed = rf"#{{1,6}}[ \t]*[*_]*\[?[ \t]*(?:{_NUM_SECTION_PREFIX})?{core_pat}[ \t]*\]?[*_]*{trail}"
    b_delim = rf"(?:[*_]{{1,2}}\[?[ \t]*|\[[ \t]*)(?:{_NUM_SECTION_PREFIX})?{core_pat}(?:[ \t]*\][*_]{{0,2}}|[*_]{{1,2}}){trail}"
    b_bare = rf"(?:{_NUM_SECTION_PREFIX})?{core_pat}[ \t]*:?"
    return re.compile(rf"(?im)^[ \t]*(?:{b_hashed}|{b_delim}|{b_bare})[ \t]*$")


def find_section_span(text: str, section_spec: str):
    """Find ``(heading_start, heading_end, body_end, level)`` for ``section_spec`` in ``text``.

    Returns ``None`` if no matching section heading exists. ``body_end`` extends to
    the next Markdown heading of equal/higher level (or next bracketed marker when
    un-hashed), ignoring headings inside fenced code blocks.
    """
    pat = section_heading_regex(section_spec)
    m = pat.search(text)
    if not m:
        return None
    line_text = text[m.start():m.end()]
    hm = re.match(r"^[ \t]*(#{1,6})(?=[ \t]|$)", line_text)
    level = len(hm.group(1)) if hm else None
    tail = text[m.end():]
    body_end = len(text)
    in_fence = False
    for lm in re.finditer(r"([^\n]*)", tail):
        ln = lm.group(1)
        if not ln:
            continue
        stripped = ln.lstrip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        hm2 = re.match(r"^[ \t]*(#{1,6})[ \t]+\S", ln)
        if hm2:
            if level is None or len(hm2.group(1)) <= level:
                body_end = m.end() + lm.start()
                break
        elif level is None and re.match(r"^[ \t]*\[[^\]\n]{1,120}\](?:[ \t]*[:\-–—].*)?[ \t]*$", ln):
            body_end = m.end() + lm.start()
            break
    return m.start(), m.end(), body_end, level


class ArtifactError(Exception):
    """An artifact operation was refused."""


class ArtifactManager:
    def __init__(self, root, max_count=16, max_bytes=131072, max_total_bytes=524288,
                 per_artifact_max_bytes=None, max_history_per_artifact=16):
        root = Path(root)
        if root.is_symlink():
            raise ArtifactError("workspace root must not be a symlink")
        root.mkdir(parents=True, exist_ok=True)
        self.root = root.resolve()
        self.max_count, self.max_bytes, self.max_total = max_count, max_bytes, max_total_bytes
        self.max_history_per_artifact = max(1, int(max_history_per_artifact))
        self.per_artifact_max_bytes = {}
        if per_artifact_max_bytes:
            for k, v in per_artifact_max_bytes.items():
                self.per_artifact_max_bytes[self.normalize(k)] = int(v)
        self.versions = {}   # file name -> version (only artifacts created through this manager exist)
        self.seen = {}       # (agent, file name) -> last version that agent read or wrote
        self.log = []        # dicts: turn, agent, op, name, version, ok, detail
        self.history = {}    # file name -> bounded list of version snapshots

    def limit_for(self, fname):
        return int(self.per_artifact_max_bytes.get(fname, self.max_bytes))

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

    def _read_file(self, path, max_bytes=None):
        cap = max_bytes if max_bytes is not None else max([self.max_bytes, *self.per_artifact_max_bytes.values()])
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(fd, "rb") as f:
            return f.read(cap + 1).decode("utf-8", "replace")

    # ── public operations ──
    def names(self):
        return sorted(self.versions)

    def listing(self):
        return [(f, self.versions[f], os.lstat(self.root / f).st_size) for f in self.names()]

    def _record_history(self, fname, version, turn, agent, op, content, prev_size_bytes):
        encoded = content.encode("utf-8")
        size_bytes = len(encoded)
        delta_bytes = size_bytes - int(prev_size_bytes)
        sha = hashlib.sha256(encoded).hexdigest()
        hist = self.history.setdefault(fname, [])
        hist.append({
            "version": int(version),
            "turn": int(turn),
            "agent": str(agent),
            "op": str(op),
            "size_bytes": size_bytes,
            "delta_bytes": delta_bytes,
            "content": content,
            "sha256": sha,
        })
        if len(hist) > self.max_history_per_artifact:
            del hist[:-self.max_history_per_artifact]

    def recent_history_summary(self, name, limit=4) -> str:
        """Return a concise 'v3(6778B) -> v4(12572B, +5794B)' trail when >= 2 versions exist."""
        try:
            fname = self.normalize(name)
        except ArtifactError:
            return ""
        hist = self.history.get(fname, [])
        if len(hist) <= 1:
            return ""
        tail = hist[-max(2, int(limit)):]
        parts = []
        for idx, entry in enumerate(tail):
            ver = entry["version"]
            sz = entry["size_bytes"]
            if idx == 0:
                parts.append(f"v{ver}({sz}B)")
            else:
                parts.append(f"v{ver}({sz}B, {entry['delta_bytes']:+d}B)")
        return " -> ".join(parts)

    def snapshot(self):
        files = {}
        for name in self.names():
            content = self._read_file(self._path(name))
            files[name] = dict(content=content, version=self.versions[name],
                               sha256=hashlib.sha256(content.encode("utf-8")).hexdigest())
        history = {name: [dict(e) for e in self.history.get(name, [])] for name in self.names()}
        return dict(files=files, seen=[[agent, name, version]
                                      for (agent, name), version in self.seen.items()],
                    log=self.log[:], history=history)

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
            if len(encoded) > self.limit_for(name):
                raise ArtifactError("saved artifact exceeds size limit")
            total_bytes += len(encoded)
        if total_bytes > self.max_total:
            raise ArtifactError("saved workspace exceeds total size limit")
        seen = {(agent, name): version for agent, name, version in snapshot["seen"]}
        for (_, name), version in seen.items():
            if name not in files or not 1 <= version <= files[name]["version"]:
                raise ArtifactError("invalid saved artifact read version")

        raw_history = snapshot.get("history")
        restored_history = {}
        if raw_history is None:
            for name, saved in files.items():
                encoded = saved["content"].encode("utf-8")
                restored_history[name] = [{
                    "version": saved["version"],
                    "turn": 0,
                    "agent": "system",
                    "op": "restore",
                    "size_bytes": len(encoded),
                    "delta_bytes": len(encoded),
                    "content": saved["content"],
                    "sha256": saved["sha256"],
                }]
        else:
            if not isinstance(raw_history, dict):
                raise ArtifactError("invalid saved artifact history")
            for name, entries in raw_history.items():
                if name not in files or not isinstance(entries, list):
                    raise ArtifactError(f"invalid saved artifact history for {name}")
                clean_entries = []
                for entry in entries:
                    if not isinstance(entry, dict):
                        raise ArtifactError(f"invalid saved artifact history entry for {name}")
                    ver = entry.get("version")
                    content = entry.get("content")
                    sha = entry.get("sha256")
                    if not isinstance(ver, int) or ver < 1 or not isinstance(content, str):
                        raise ArtifactError(f"invalid saved artifact history metadata for {name}")
                    encoded = content.encode("utf-8")
                    if hashlib.sha256(encoded).hexdigest() != sha:
                        raise ArtifactError(f"saved artifact history checksum mismatch: {name}")
                    clean_entries.append({
                        "version": ver,
                        "turn": int(entry.get("turn", 0)),
                        "agent": str(entry.get("agent", "")),
                        "op": str(entry.get("op", "write")),
                        "size_bytes": int(entry.get("size_bytes", len(encoded))),
                        "delta_bytes": int(entry.get("delta_bytes", 0)),
                        "content": content,
                        "sha256": sha,
                    })
                restored_history[name] = clean_entries
            for name, saved in files.items():
                if not restored_history.get(name):
                    encoded = saved["content"].encode("utf-8")
                    restored_history[name] = [{
                        "version": saved["version"],
                        "turn": 0,
                        "agent": "system",
                        "op": "restore",
                        "size_bytes": len(encoded),
                        "delta_bytes": len(encoded),
                        "content": saved["content"],
                        "sha256": saved["sha256"],
                    }]

        for name, saved in files.items():
            self._atomic_write(self._path(name), saved["content"])
        self.versions = {name: saved["version"] for name, saved in files.items()}
        self.seen = seen
        self.log = list(snapshot["log"])
        self.history = restored_history

    def read(self, agent, name, turn=0, mark_seen=True):
        try:
            fname = self.normalize(name)
            if fname not in self.versions:
                raise ArtifactError(f"no artifact named {fname}")
            text = self._read_file(self._path(fname))
        except ArtifactError as e:
            self.log.append(dict(turn=turn, agent=agent, op="read", name=str(name)[:60], version=None, ok=False, detail=str(e)))
            raise
        if mark_seen:
            self.seen[(agent, fname)] = self.versions[fname]
        self.log.append(dict(turn=turn, agent=agent, op="read", name=fname, version=self.versions[fname], ok=True, detail=""))
        return fname, self.versions[fname], text

    def write(self, agent, name, mode, content, expected_version=None, turn=0, attribution=True,
              section=None, force=False, min_replace_ratio=0.0):
        try:
            fname, version, size = self._write(
                agent, name, mode, content, expected_version, turn, attribution,
                section=section, force=force, min_replace_ratio=min_replace_ratio,
            )
        except ArtifactError as e:
            self.log.append(dict(turn=turn, agent=agent, op=str(mode)[:16], name=str(name)[:60], version=None, ok=False, detail=str(e)))
            raise
        self.log.append(dict(turn=turn, agent=agent, op=mode, name=fname, version=version, ok=True,
                             detail=f"{len(content)} chars -> {size}B"))
        return fname, version

    def _write(self, agent, name, mode, content, expected_version, turn, attribution,
               section=None, force=False, min_replace_ratio=0.0):
        if mode not in ("create", "append", "replace", "replace_section"):
            raise ArtifactError("mode must be create, append, replace or replace_section")
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
        if mode in ("replace", "replace_section") and exists and expected_version is None and self.seen.get((agent, fname)) != current:
            raise ArtifactError(f"{fname} changed since you last saw it (now v{current}); read it before replacing")
        current_size = os.lstat(path).st_size if exists else 0
        req_size = len(content.encode("utf-8"))
        art_limit = self.limit_for(fname)
        short_name = fname.removesuffix(".md")

        if mode == "replace_section":
            if not isinstance(section, str) or not section.strip():
                raise ArtifactError("replace_section requires a target section heading")
            sec_spec = CONTROL_CHARS.sub("", section).strip()
            body_clean = content.strip()
            sec_pat = section_heading_regex(sec_spec)
            first_line = next((ln for ln in body_clean.splitlines() if ln.strip()), "")
            has_heading = bool(sec_pat.match(first_line))
            if exists:
                old = self._read_file(path, max_bytes=art_limit)
                span = find_section_span(old, sec_spec)
                if span:
                    h_start, h_end, b_end, _ = span
                    if has_heading:
                        replacement_block = body_clean
                    else:
                        replacement_block = old[h_start:h_end].rstrip("\n") + "\n\n" + body_clean
                    parts = [old[:h_start].rstrip("\n"), replacement_block, old[b_end:].lstrip("\n")]
                    new = "\n\n".join(p for p in parts if p) + "\n"
                else:
                    heading_line = sec_spec if sec_spec.startswith(("#", "[")) else f"## {sec_spec}"
                    add_block = body_clean if has_heading else f"{heading_line}\n\n{body_clean}"
                    new = old.rstrip("\n") + "\n\n" + add_block + "\n"
            else:
                heading_line = sec_spec if sec_spec.startswith(("#", "[")) else f"## {sec_spec}"
                new = (body_clean if has_heading else f"{heading_line}\n\n{body_clean}") + "\n"
        elif mode == "append" and exists:
            old = self._read_file(path, max_bytes=art_limit)
            note = f"\n\n— *{agent}, turn {turn}*" if attribution else ""
            new = old.rstrip("\n") + "\n\n" + content.strip() + note + "\n"
        else:
            new = content.strip() + "\n"

        size = len(new.encode("utf-8"))
        if mode == "replace" and exists and not force:
            if current_size >= 500 and is_placeholder_stub(content):
                raise ArtifactError(
                    f"{fname} replace rejected: content looks like a template/placeholder stub rather than full text. "
                    f"Use @SECTION {short_name} | <heading>: <content> to edit one section, "
                    f"or @FORCE_REPLACE {short_name}: <content> if intentional."
                )
            if min_replace_ratio and current_size >= 1500 and current_size < int(art_limit * 0.85):
                min_allowed = max(300, int(current_size * float(min_replace_ratio)))
                if size < min_allowed:
                    raise ArtifactError(
                        f"{fname} replace rejected: would shrink artifact from {current_size}B to {size}B "
                        f"(< {int(float(min_replace_ratio) * 100)}% of current size). "
                        f"Use @SECTION {short_name} | <heading>: <content> to update one section without overwriting peers' work, "
                        f"@APPEND {short_name}: ... to add content, or @FORCE_REPLACE {short_name}: <content> if intentional."
                    )

        if size > art_limit:
            raise ArtifactError(
                f"{fname} write rejected: current={current_size}B, update={req_size}B, "
                f"resulting={size}B exceeds limit={art_limit}B. "
                f"Use @REPLACE {short_name}: <consolidated content> to overwrite/trim, "
                f"or save working drafts in a separate artifact."
            )
        other_bytes = self._total_bytes(exclude=fname)
        if other_bytes + size > self.max_total:
            raise ArtifactError(
                f"workspace size limit reached: other_artifacts={other_bytes}B + {fname}={size}B "
                f"({other_bytes + size}B) exceeds max_workspace_bytes={self.max_total}B. "
                f"Use @REPLACE {short_name}: <consolidated content> to trim."
            )
        self._atomic_write(path, new)
        new_ver = current + 1
        self.versions[fname] = new_ver
        self.seen[(agent, fname)] = new_ver
        self._record_history(fname, new_ver, turn, agent, mode, new, current_size)
        return fname, new_ver, size

    def revert(self, agent, name, target_version, turn=0):
        """Restore an artifact's content from a saved historical version without deleting history."""
        try:
            fname, new_ver, target_ver, size, delta = self._revert(agent, name, target_version, turn)
        except ArtifactError as e:
            self.log.append(dict(turn=turn, agent=agent, op="revert", name=str(name)[:60], version=None, ok=False, detail=str(e)))
            raise
        self.log.append(dict(
            turn=turn, agent=agent, op="revert", name=fname, version=new_ver, ok=True,
            detail=f"reverted to v{target_ver} -> {size}B ({delta:+d}B)",
        ))
        return fname, new_ver, target_ver, size, delta

    def _revert(self, agent, name, target_version, turn):
        fname = self.normalize(name)
        if fname not in self.versions:
            raise ArtifactError(f"no artifact named {fname}")
        if isinstance(target_version, bool):
            raise ArtifactError(f"invalid target version {target_version!r} for {fname}")
        target_ver = None
        if isinstance(target_version, int):
            target_ver = target_version
        elif isinstance(target_version, str):
            cleaned = re.sub(r"^[vV]+", "", target_version.strip().strip("[]()<>\"'`,.;:"))
            if re.fullmatch(r"\d+", cleaned):
                target_ver = int(cleaned)
        if target_ver is None or target_ver < 1:
            raise ArtifactError(f"invalid target version {target_version!r} for {fname}")

        current = self.versions[fname]
        if target_ver == current:
            raise ArtifactError(f"{fname} is already at v{current}")

        hist = self.history.get(fname, [])
        entry = next((e for e in hist if e.get("version") == target_ver), None)
        if entry is None:
            avail = ", ".join(f"v{e['version']}" for e in hist) or "none"
            raise ArtifactError(f"{fname} has no saved history for v{target_ver} (available: {avail})")

        restored_text = str(entry["content"])
        size = len(restored_text.encode("utf-8"))
        art_limit = self.limit_for(fname)
        if size > art_limit:
            raise ArtifactError(f"{fname} revert rejected: v{target_ver} ({size}B) exceeds limit={art_limit}B")
        other_bytes = self._total_bytes(exclude=fname)
        if other_bytes + size > self.max_total:
            raise ArtifactError(
                f"workspace size limit reached: other_artifacts={other_bytes}B + {fname}={size}B "
                f"exceeds max_workspace_bytes={self.max_total}B"
            )

        path = self._path(fname)
        current_size = os.lstat(path).st_size if path.exists() else 0
        delta = size - current_size
        self._atomic_write(path, restored_text)
        new_ver = current + 1
        self.versions[fname] = new_ver
        self.seen[(agent, fname)] = new_ver
        self._record_history(fname, new_ver, turn, agent, "revert", restored_text, current_size)
        return fname, new_ver, target_ver, size, delta


DEFAULT_PLACEHOLDER_PATTERNS = (
    r"\.\.\.\s*\(full\s+(?:lyrics|text|content|track|draft)\b",
    r"\.\.\.\s*\(\s*insert\b[^)]*\)",
    r"\[\s*(?:TBD|TODO|PLACEHOLDER|INSERT\b[^\]]*)\s*\]",
    r"(?m)^\s*(?:TBD|TODO)\s*[:\-]?\s*$",
    r"Track Title:\s*TBD\b",
)


def is_placeholder_stub(text: str) -> bool:
    """Return True if ``text`` is an unfilled template or stub replacement rather than full content."""
    s = str(text or "").strip()
    if not s:
        return True
    if re.fullmatch(r"<[^<>\n]{1,160}>", s):
        return True
    if len(s) < 250:
        for pat in DEFAULT_PLACEHOLDER_PATTERNS:
            if re.search(pat, s, re.I):
                return True
        if re.fullmatch(r"\[(?:END|DONE|COMPLETE|FINAL)[^\]]*\]", s, re.I):
            return True
    return False


def validate_artifact(manager: ArtifactManager, spec: dict):
    """Validate a deliverable artifact against configurable completeness rules.

    Returns (ok: bool, issues: list[str], details: dict).
    """
    if not isinstance(spec, dict) or not spec.get("artifact"):
        return True, [], {}

    raw_name = str(spec["artifact"])
    try:
        fname = manager.normalize(raw_name)
    except ArtifactError as exc:
        return False, [str(exc)], {"artifact": raw_name}

    issues = []
    if fname not in manager.versions:
        return False, [f"Required deliverable {fname} does not exist yet."], {"artifact": fname, "exists": False}

    version = manager.versions[fname]
    if spec.get("require_modified", True) and version <= 1:
        issues.append(f"{fname} is still at v{version} (unmodified seed); write the actual deliverable before ending.")

    path = manager._path(fname)
    text = manager._read_file(path, max_bytes=manager.limit_for(fname))
    size_bytes = len(text.encode("utf-8"))
    max_allowed = int(spec.get("max_bytes") or manager.limit_for(fname))
    if size_bytes > max_allowed:
        issues.append(f"{fname} is {size_bytes}B, exceeding configured deliverable limit {max_allowed}B.")

    min_bytes = int(spec.get("min_bytes") or 0)
    if min_bytes and size_bytes < min_bytes:
        issues.append(f"{fname} is too short ({size_bytes}B < required {min_bytes}B).")

    # Placeholder checks
    disallow = spec.get("disallow_placeholders", True)
    patterns = list(DEFAULT_PLACEHOLDER_PATTERNS) if disallow is True else (list(disallow) if isinstance(disallow, (list, tuple)) else [])
    for pat in patterns:
        m = re.search(pat, text, re.I)
        if m:
            issues.append(f"{fname} contains unfinished placeholder text matching {m.group(0)!r}; replace with full content.")

    # Required sections check (case-insensitive header/marker match with non-empty body)
    req_sections = [str(s) for s in (spec.get("required_sections") or []) if str(s).strip()]
    missing_sections = []
    empty_sections = []
    if req_sections:
        positions = []
        for sec in req_sections:
            pat = section_heading_regex(sec)
            m = pat.search(text)
            if not m:
                missing_sections.append(sec)
            else:
                positions.append((m.start(), m.end(), sec))
        positions.sort()
        for idx, (_, end_pos, sec) in enumerate(positions):
            next_start = positions[idx + 1][0] if idx + 1 < len(positions) else len(text)
            body = text[end_pos:next_start].strip()
            # Ignore pure stage-direction/metadata-only 1-liners < 20 chars
            if len(body) < 20:
                empty_sections.append(sec)

    if missing_sections:
        issues.append(f"{fname} is missing required section(s): {', '.join(missing_sections)}.")
    if empty_sections:
        issues.append(f"{fname} has empty/stub section(s): {', '.join(empty_sections)}.")

    details = {
        "artifact": fname,
        "version": version,
        "size_bytes": size_bytes,
        "min_bytes": min_bytes,
        "required_sections_count": len(req_sections),
        "matched_sections_count": len(req_sections) - len(missing_sections),
        "missing_sections": missing_sections,
        "empty_sections": empty_sections,
    }
    return (len(issues) == 0), issues, details
