"""Markdown artifact workspace: the only place agent actions can cause durable writes.

Agents never supply paths. They name an artifact ("world", "world.md" or "artifacts/world.md"); the name is
validated against a strict pattern and mapped to <workspace>/<name>.md. Symlinks, traversal, non-Markdown
files, oversized content and stale overwrites are rejected. Content is stored as data and never interpreted.
"""
import difflib
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

    Prefers the shallowest non-fenced matching heading so an accidentally nested
    subsection heading never shadows a top-level section heading. ``body_end`` extends
    to the next Markdown heading of equal/higher level (or next bracketed marker when
    un-hashed), ignoring headings inside fenced code blocks.
    """
    pat = section_heading_regex(section_spec)
    candidates = []
    in_fence = False
    pos = 0
    for raw_ln in str(text or "").splitlines(keepends=True):
        ln_no_nl = raw_ln.rstrip("\r\n")
        if ln_no_nl.lstrip().startswith(("```", "~~~")):
            in_fence = not in_fence
        elif not in_fence and pat.match(ln_no_nl):
            hm = re.match(r"^[ \t]*(#{1,6})(?=[ \t]|$)", ln_no_nl)
            match_level = len(hm.group(1)) if hm else 99
            real_level = match_level if hm else None
            candidates.append((match_level, pos, pos + len(ln_no_nl), real_level))
        pos += len(raw_ln)
    if not candidates:
        return None
    candidates.sort(key=lambda c: (c[0], c[1]))
    _, h_start, h_end, level = candidates[0]
    tail = text[h_end:]
    body_end = len(text)
    in_fence = False
    for lm in re.finditer(r"([^\n]*)", tail):
        ln = lm.group(1)
        if not ln:
            continue
        stripped = ln.lstrip()
        if stripped.startswith(("```", "~~~")):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        hm2 = re.match(r"^[ \t]*(#{1,6})[ \t]+\S", ln)
        if hm2:
            if level is None or len(hm2.group(1)) <= level:
                body_end = h_end + lm.start()
                break
        elif level is None and re.match(r"^[ \t]*\[[^\]\n]{1,120}\](?:[ \t]*[:\-–—].*)?[ \t]*$", ln):
            body_end = h_end + lm.start()
            break
    return h_start, h_end, body_end, level


def normalize_section_body_headings(body: str, parent_level: int) -> str:
    """Demote internal ATX headings inside a section body so they nest strictly below ``parent_level``."""
    if not body or parent_level < 1:
        return body
    lines = body.splitlines()
    in_fence = False
    matches = []  # (line_idx, indent, hashes, rest)
    for idx, ln in enumerate(lines):
        if ln.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        hm = re.match(r"^([ \t]*)(#{1,6})([ \t]+\S.*)$", ln)
        if hm:
            matches.append((idx, hm.group(1), hm.group(2), hm.group(3)))
    if not matches:
        return body
    min_lvl = min(len(h) for _, _, h, _ in matches)
    if min_lvl > parent_level:
        return body
    shift = (parent_level + 1) - min_lvl
    for idx, indent, hashes, rest in matches:
        new_hashes = "#" * min(6, len(hashes) + shift)
        lines[idx] = f"{indent}{new_hashes}{rest}"
    trailing_nl = "\n" if body.endswith("\n") else ""
    return "\n".join(lines) + trailing_nl


_PROTECTED_NL_SPAN_RE = re.compile(
    r"(?:```[\s\S]*?```"
    r"|~~~[\s\S]*?~~~"
    r"|``[^\n]+?``"
    r"|`[^`\n]+`"
    r"|\\begin\{(?P<math_env>(?:equation|align|gather|multline|eqnarray|math|displaymath|flalign|alignat)\*?)\}[\s\S]*?\\end\{(?P=math_env)\}"
    r"|\$\$[\s\S]*?\$\$"
    r"|\\\[[\s\S]*?\\\]"
    r"|\\\([\s\S]*?\\\)"
    r"|(?<!\\)\$(?!\$)(?:\\.|[^$\\\n])+\$)"
)

_SINGLE_NL_BOUNDARY_RE = re.compile(
    r"(?<!\\)\\n(?=(?:[ \t]*(?:#{1,6}[ \t]|[-*+][ \t]|\d+\.[ \t]|>[ \t]|\*\*|__|\$)|[ \t]*$))"
)


def normalize_escaped_newlines(text: str) -> str:
    """Convert unambiguous literal ``\\n`` line breaks into real newlines without touching code or LaTeX."""
    if not text or "\\n" not in text:
        return text

    def _convert_prose(seg: str) -> str:
        if "\\n" not in seg:
            return seg
        # Multi-newline escape runs (\n\n...) not immediately preceding lowercase macro letters (\neq/\nu)
        seg = re.sub(
            r"(?<!\\)((?:\\n){2,})(?![a-z])",
            lambda m: "\n" * (len(m.group(1)) // 2),
            seg,
        )
        # Single escaped newline directly before Markdown structure or end-of-segment
        seg = _SINGLE_NL_BOUNDARY_RE.sub("\n", seg)
        return seg

    out = []
    pos = 0
    for m in _PROTECTED_NL_SPAN_RE.finditer(text):
        if m.start() > pos:
            out.append(_convert_prose(text[pos:m.start()]))
        out.append(m.group(0))
        pos = m.end()
    if pos < len(text):
        out.append(_convert_prose(text[pos:]))
    return "".join(out)


def _extract_atx_headings(text: str, min_level: int = 3, max_level: int = 6):
    """Return list of ``(normalized_key, raw_heading)`` for non-fenced ATX headings in ``min_level..max_level``."""
    out = []
    in_fence = False
    for ln in str(text or "").splitlines():
        if ln.lstrip().startswith(("```", "~~~")):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        hm = re.match(r"^[ \t]*(#{1,6})[ \t]+(.+?)[ \t]*$", ln)
        if not hm:
            continue
        lvl = len(hm.group(1))
        if not (min_level <= lvl <= max_level):
            continue
        raw_h = hm.group(2).strip()
        norm_h = re.sub(r"\s+", " ", raw_h.strip("[]*_")).strip().lower()
        if norm_h:
            out.append((norm_h, raw_h))
    return out


def _find_embedded_other_section_heading(body_text: str, other_text: str, max_level: int):
    """Return a heading in ``body_text`` that collides with a sibling section heading in ``other_text``."""
    if not body_text or not other_text or max_level < 1:
        return None
    other_pats = []
    seen_norm = set()
    for norm_h, raw_h in _extract_atx_headings(other_text, min_level=1, max_level=max_level):
        if norm_h not in seen_norm:
            seen_norm.add(norm_h)
            other_pats.append((norm_h, section_heading_regex(raw_h)))
    if not other_pats:
        return None

    in_fence = False
    for ln in str(body_text or "").splitlines():
        if ln.lstrip().startswith(("```", "~~~")):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        hm = re.match(r"^[ \t]*(#{1,6})[ \t]+(.+?)[ \t]*$", ln)
        if not hm:
            continue
        raw_h = hm.group(2).strip()
        norm_h = re.sub(r"\s+", " ", raw_h.strip("[]*_")).strip().lower()
        for other_norm, pat in other_pats:
            if norm_h == other_norm or pat.match(ln):
                return raw_h
    return None


def _strip_fences_and_normalize_prose(text: str) -> str:
    """Strip fenced code blocks and light Markdown formatting for conservative duplicate comparison."""
    lines = []
    in_fence = False
    for ln in str(text or "").splitlines():
        if ln.lstrip().startswith(("```", "~~~")):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        clean = re.sub(r"^[ \t]*#{1,6}[ \t]+", "", ln)
        clean = clean.replace("**", "").replace("__", "")
        lines.append(clean)
    return re.sub(r"\s+", " ", "\n".join(lines)).strip().lower()


def _word_ngrams(norm_text: str, n: int = 4):
    """Return the set of consecutive alphanumeric word n-grams in ``norm_text``."""
    words = re.findall(r"[a-z0-9]+", norm_text)
    if len(words) < n:
        return set()
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def _split_comparison_blocks(text: str, min_chars: int = 180):
    """Extract normalized subsection and substantial paragraph blocks from non-fenced Markdown."""
    blocks = []
    seen_blocks = set()
    in_fence = False
    clean_lines = []
    for ln in str(text or "").splitlines():
        if ln.lstrip().startswith(("```", "~~~")):
            in_fence = not in_fence
            continue
        if not in_fence:
            clean_lines.append(ln)
    clean_text = "\n".join(clean_lines)

    def _add(raw_chunk: str):
        norm = _strip_fences_and_normalize_prose(raw_chunk)
        if len(norm) >= min_chars and norm not in seen_blocks:
            seen_blocks.add(norm)
            blocks.append(norm)

    # Subsection blocks (###..######)
    sub_parts = re.split(r"(?=^[ \t]*#{3,6}[ \t]+\S)", clean_text, flags=re.M)
    for part in sub_parts:
        _add(part)

    # Substantial paragraphs and 2-4 paragraph windows
    paras = [p.strip() for p in re.split(r"\n[ \t]*\n", clean_text) if p.strip()]
    for p in paras:
        _add(p)
    for win in (2, 3, 4):
        if len(paras) >= win:
            for i in range(len(paras) - win + 1):
                _add("\n\n".join(paras[i:i + win]))
    return blocks


_DUP_MIN_EXACT_CHARS = 140
_DUP_MIN_FUZZY_CHARS = 180
_DUP_MIN_NGRAMS = 20
_DUP_NGRAM_THRESHOLD = 0.80
_DUP_SEQ_LEN_RATIO = 0.65
_DUP_SEQ_MIN_NGRAMS = 15
_DUP_SEQ_RATIO = 0.82


def _blocks_near_identical(norm_a: str, norm_b: str, ng_a=None, ng_b=None) -> bool:
    """Return True if two normalized prose blocks are exact or near-identical duplicates."""
    if not norm_a or not norm_b:
        return False
    if len(norm_a) >= _DUP_MIN_EXACT_CHARS and (norm_a == norm_b or norm_a in norm_b):
        return True
    if min(len(norm_a), len(norm_b)) < _DUP_MIN_FUZZY_CHARS:
        return False
    ng_a = ng_a if ng_a is not None else _word_ngrams(norm_a, 4)
    ng_b = ng_b if ng_b is not None else _word_ngrams(norm_b, 4)
    if len(ng_a) >= _DUP_MIN_NGRAMS and ng_b:
        if (len(ng_a & ng_b) / len(ng_a)) >= _DUP_NGRAM_THRESHOLD:
            return True
    if len(ng_a) >= _DUP_SEQ_MIN_NGRAMS:
        ratio_len = min(len(norm_a), len(norm_b)) / max(len(norm_a), len(norm_b))
        if ratio_len >= _DUP_SEQ_LEN_RATIO and difflib.SequenceMatcher(None, norm_a, norm_b).ratio() >= _DUP_SEQ_RATIO:
            return True
    return False


def find_duplicate_append_reason(
    existing_text: str,
    candidate_text: str,
    check_headings: bool = True,
    min_heading_level: int = 3,
):
    """Return a rejection reason if appending ``candidate_text`` would duplicate existing content, else ``None``.

    Strictly rejects exact duplicate subheadings and exact/near-identical repeated blocks
    while staying conservative for shorter formulas, shared definitions, and citations.
    """
    if not candidate_text or not existing_text:
        return None

    if check_headings:
        existing_h = {k for k, _ in _extract_atx_headings(existing_text, min_level=min_heading_level, max_level=6)}
        seen_cand = set()
        for norm_h, raw_h in _extract_atx_headings(candidate_text, min_level=min_heading_level, max_level=6):
            if norm_h in existing_h:
                return f"subheading {raw_h!r} already exists in this section"
            if norm_h in seen_cand:
                return f"subheading {raw_h!r} is repeated within appended content"
            seen_cand.add(norm_h)

    cand_norm = _strip_fences_and_normalize_prose(candidate_text)
    exist_norm = _strip_fences_and_normalize_prose(existing_text)
    if not cand_norm or not exist_norm:
        return None

    if len(cand_norm) >= _DUP_MIN_EXACT_CHARS and cand_norm in exist_norm:
        return "appended content already exists verbatim in this section"

    if len(cand_norm) >= _DUP_MIN_FUZZY_CHARS:
        cand_4 = _word_ngrams(cand_norm, 4)
        exist_4 = _word_ngrams(exist_norm, 4)
        if len(cand_4) >= _DUP_MIN_NGRAMS and exist_4:
            containment = len(cand_4 & exist_4) / len(cand_4)
            if containment >= _DUP_NGRAM_THRESHOLD:
                pct = int(round(containment * 100))
                return f"appended content heavily overlaps existing text ({pct}% 4-gram overlap)"

        exist_blocks = _split_comparison_blocks(existing_text, min_chars=_DUP_MIN_FUZZY_CHARS)
        cand_units = [cand_norm]
        cand_subs = re.split(r"(?=^[ \t]*#{3,6}[ \t]+\S)", candidate_text, flags=re.M)
        if len(cand_subs) > 1:
            for sub in cand_subs:
                sn = _strip_fences_and_normalize_prose(sub)
                if len(sn) >= _DUP_MIN_FUZZY_CHARS and sn not in cand_units:
                    cand_units.append(sn)
        for cu in cand_units:
            cu_4 = _word_ngrams(cu, 4)
            for eb in exist_blocks:
                if _blocks_near_identical(cu, eb, ng_a=cu_4):
                    return "appended block is near-identical to an existing block in this section"

    return None


def find_internal_subsection_duplicates(text: str):
    """Detect duplicate ``###``+ subheadings or near-identical repeated blocks inside each ``##`` section."""
    if not text:
        return {}

    lines = text.splitlines()
    sections = []  # list of (section_title, body_text)
    cur_title = "[document]"
    cur_lines = []
    in_fence = False
    saw_h2 = False
    for ln in lines:
        if ln.lstrip().startswith(("```", "~~~")):
            in_fence = not in_fence
            cur_lines.append(ln)
            continue
        if not in_fence:
            hm2 = re.match(r"^[ \t]*##[ \t]+(.+?)[ \t]*$", ln)
            if hm2:
                if cur_lines and (saw_h2 or any(x.strip() for x in cur_lines)):
                    sections.append((cur_title, "\n".join(cur_lines)))
                cur_title = hm2.group(1).strip()
                cur_lines = []
                saw_h2 = True
                continue
        cur_lines.append(ln)
    if cur_lines and (saw_h2 or any(x.strip() for x in cur_lines)):
        sections.append((cur_title, "\n".join(cur_lines)))

    findings = {}
    for sec_title, sec_body in sections:
        h_counts = {}
        h_labels = {}
        for norm_h, raw_h in _extract_atx_headings(sec_body, min_level=3, max_level=6):
            h_counts[norm_h] = h_counts.get(norm_h, 0) + 1
            h_labels.setdefault(norm_h, raw_h)
        dup_headings = {h_labels[k]: c for k, c in h_counts.items() if c >= 2}

        # Strip code fences and extract non-overlapping chunks (subsections if present, plus >=180-char paragraphs)
        clean_lines = []
        in_f = False
        for ln in sec_body.splitlines():
            if ln.lstrip().startswith(("```", "~~~")):
                in_f = not in_f
                continue
            if not in_f:
                clean_lines.append(ln)
        clean_body = "\n".join(clean_lines)

        chunks = []
        raw_subs = re.split(r"(?=^[ \t]*#{3,6}[ \t]+\S)", clean_body, flags=re.M)
        if len(raw_subs) > 1:
            chunks = raw_subs
        else:
            chunks = [p.strip() for p in re.split(r"\n[ \t]*\n", clean_body) if p.strip()]

        norm_chunks = []
        for ch in chunks:
            nc = _strip_fences_and_normalize_prose(ch)
            if len(nc) >= _DUP_MIN_EXACT_CHARS:
                norm_chunks.append((nc, _word_ngrams(nc, 4)))

        repeated_blocks = 0
        for j in range(1, len(norm_chunks)):
            nj, ng_j = norm_chunks[j]
            if any(_blocks_near_identical(nj, ni, ng_a=ng_j, ng_b=ng_i) for ni, ng_i in norm_chunks[:j]):
                repeated_blocks += 1

        if dup_headings or repeated_blocks:
            entry = {}
            if dup_headings:
                entry["duplicate_subheadings"] = dup_headings
            if repeated_blocks:
                entry["repeated_blocks"] = repeated_blocks
            findings[sec_title] = entry
    return findings


def _strip_trailing_attribution(text: str, agent: str) -> str:
    """Remove redundant trailing agent attribution line(s) before appending harness attribution."""
    if not text or not agent:
        return text
    pat = re.compile(
        rf"(?:\n+[ \t]*[*_]*(?:—|–|--)[ \t]*[*_]*{re.escape(str(agent))}[*_]*(?:[ \t]*,[ \t]*turn[ \t]+\d+)?[*_]*[ \t]*)+$",
        re.I,
    )
    stripped = pat.sub("", text.rstrip()).rstrip()
    return stripped if stripped else text.strip()


class ArtifactError(Exception):
    """An artifact operation was refused."""


class ArtifactManager:
    def __init__(self, root, max_count=16, max_bytes=131072, max_total_bytes=524288,
                 per_artifact_max_bytes=None, max_history_per_artifact=24):
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
        log_entry = dict(
            turn=turn, agent=agent, op=mode, name=fname, version=version, ok=True,
            detail=f"{len(content)} chars -> {size}B",
        )
        if mode in ("replace_section", "append_section") and section:
            log_entry["section"] = CONTROL_CHARS.sub("", str(section)).strip()
        self.log.append(log_entry)
        return fname, version

    def _write(self, agent, name, mode, content, expected_version, turn, attribution,
               section=None, force=False, min_replace_ratio=0.0):
        if mode not in ("create", "append", "replace", "replace_section", "append_section"):
            raise ArtifactError("mode must be create, append, replace, replace_section or append_section")
        if not isinstance(content, str) or not content.strip():
            raise ArtifactError("artifact content must be non-empty text")
        content = CONTROL_CHARS.sub("", content)
        content = normalize_escaped_newlines(content)
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

        if mode in ("replace_section", "append_section"):
            if not isinstance(section, str) or not section.strip():
                raise ArtifactError(f"{mode} requires a target section heading")
            sec_spec = CONTROL_CHARS.sub("", section).strip()
            body_clean = content.strip()
            sec_pat = section_heading_regex(sec_spec)
            raw_lines = body_clean.splitlines()
            first_idx = next((idx for idx, ln in enumerate(raw_lines) if ln.strip()), None)
            first_line = raw_lines[first_idx] if first_idx is not None else ""
            has_heading = bool(first_line and sec_pat.match(first_line))
            body_without_heading = (
                "\n".join(raw_lines[first_idx + 1:]).strip() if (has_heading and first_idx is not None) else body_clean
            )
            if exists:
                old = self._read_file(path, max_bytes=art_limit)
                span = find_section_span(old, sec_spec)
                if span:
                    h_start, h_end, b_end, level = span
                    canonical_heading = old[h_start:h_end].rstrip("\n")
                    eff_level = level if level is not None else 2
                    body_norm = normalize_section_body_headings(body_without_heading, eff_level)
                    embedded_other = _find_embedded_other_section_heading(
                        body_norm, old[:h_start] + "\n" + old[b_end:], max_level=eff_level,
                    )
                    if embedded_other:
                        raise ArtifactError(
                            f"{fname} {mode} rejected: body embeds heading {embedded_other!r} which already exists "
                            f"as another section in {fname}; update each section with its own @SECTION command "
                            f"instead of nesting it inside {sec_spec}."
                        )
                    if mode == "append_section":
                        dup_reason = find_duplicate_append_reason(
                            old[h_end:b_end],
                            body_norm,
                            check_headings=True,
                            min_heading_level=eff_level + 1,
                        )
                        if dup_reason:
                            raise ArtifactError(
                                f"{fname} append_section rejected: {dup_reason}. "
                                f"Use @SECTION {short_name} | {sec_spec}: <consolidated body> "
                                f"to edit/consolidate instead of appending a duplicate."
                            )
                        existing_block = old[h_start:b_end].rstrip("\n")
                        section_block = f"{existing_block}\n\n{body_norm}" if body_norm else existing_block
                    else:
                        if not force:
                            old_sec_bytes = len(old[h_end:b_end].strip().encode("utf-8"))
                            new_sec_bytes = len(body_norm.encode("utf-8"))
                            if old_sec_bytes >= 300 and is_placeholder_stub(body_norm):
                                raise ArtifactError(
                                    f"{fname} replace_section rejected: replacement for {sec_spec} looks like a "
                                    f"placeholder/meta-stub rather than full section content."
                                )
                            if min_replace_ratio and old_sec_bytes >= 1000:
                                min_sec_allowed = max(250, int(old_sec_bytes * float(min_replace_ratio)))
                                if new_sec_bytes < min_sec_allowed:
                                    raise ArtifactError(
                                        f"{fname} replace_section rejected: would shrink {sec_spec} from "
                                        f"{old_sec_bytes}B to {new_sec_bytes}B (< {int(float(min_replace_ratio) * 100)}% "
                                        f"of current section size); preserve verified content when consolidating or "
                                        f"use @APPEND_SECTION to add new subsections."
                                    )
                        section_block = f"{canonical_heading}\n\n{body_norm}" if body_norm else canonical_heading
                    parts = [old[:h_start].rstrip("\n"), section_block, old[b_end:].strip("\n")]
                    new = "\n\n".join(p for p in parts if p) + "\n"
                else:
                    heading_line = sec_spec if sec_spec.startswith(("#", "[")) else f"## {sec_spec}"
                    first_hdg = first_line.strip() if has_heading else heading_line
                    hm_new = re.match(r"^[ \t]*(#{1,6})(?=[ \t]|$)", first_hdg)
                    eff_level = len(hm_new.group(1)) if hm_new else 2
                    rest_norm = normalize_section_body_headings(body_without_heading, eff_level)
                    add_block = f"{first_hdg}\n\n{rest_norm}" if rest_norm else first_hdg
                    new = old.rstrip("\n") + "\n\n" + add_block + "\n"
            else:
                heading_line = sec_spec if sec_spec.startswith(("#", "[")) else f"## {sec_spec}"
                first_hdg = first_line.strip() if has_heading else heading_line
                hm_new = re.match(r"^[ \t]*(#{1,6})(?=[ \t]|$)", first_hdg)
                eff_level = len(hm_new.group(1)) if hm_new else 2
                rest_norm = normalize_section_body_headings(body_without_heading, eff_level)
                new = (f"{first_hdg}\n\n{rest_norm}" if rest_norm else first_hdg) + "\n"
        elif mode == "append" and exists:
            old = self._read_file(path, max_bytes=art_limit)
            append_body = _strip_trailing_attribution(content.strip(), agent) if attribution else content.strip()
            dup_reason = find_duplicate_append_reason(old, append_body, check_headings=False)
            if dup_reason:
                raise ArtifactError(
                    f"{fname} append rejected: {dup_reason}. "
                    f"Consolidate existing text with @SECTION or @REPLACE instead of appending a duplicate."
                )
            note = f"\n\n— *{agent}, turn {turn}*" if attribution else ""
            new = old.rstrip("\n") + "\n\n" + append_body + note + "\n"
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

SECTION_STUB_PATTERNS = (
    r"\b(?:audit|review|verification|section|draft|analysis|report|synthesis)\b[^.\n]{0,60}\bis\s+(?:currently\s+)?(?:underway|in\s+progress|pending|ongoing)\b",
    r"\b(?:to\s+be\s+(?:completed|written|added|filled|finalized|determined)|work\s+in\s+progress|coming\s+soon|will\s+be\s+(?:added|filled|completed|updated)\s+(?:later|shortly|soon|next))\b",
    r"\b(?:stub|placeholder\s+section|not\s+yet\s+(?:written|completed|populated|audited))\b",
)

REFEREE_SECTION_RE = re.compile(r"\b(?:referee|red[- ]?team|final\s+audit|verification\s+report)\b", re.I)


_META_STUB_PATTERNS = (
    r"^\(?this\s+section\s+(?:is|has\s+been|will\s+be)\s+(?:replaced|updated|expanded|populated|moved)\b",
    r"^\(?(?:see|refer\s+to)\s+(?:the\s+)?(?:content|text|section|body)\s+(?:above|below)\b",
)


def is_placeholder_stub(text: str) -> bool:
    """Return True if ``text`` is an unfilled template or stub replacement rather than full content."""
    s = str(text or "").strip()
    if not s:
        return True
    if re.fullmatch(r"<[^<>\n]{1,160}>", s):
        return True
    if len(s) < 250:
        for pat in DEFAULT_PLACEHOLDER_PATTERNS + _META_STUB_PATTERNS:
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
    warnings = []
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

    # Required sections check (case-insensitive header/marker match with non-empty body, completeness, and uniqueness)
    raw_req = [str(s) for s in (spec.get("required_sections") or []) if str(s).strip()]
    ref_explicit = [str(s) for s in (spec.get("required_referee_sections") or []) if str(s).strip()]
    req_sections = list(raw_req)
    for rs in ref_explicit:
        if rs not in req_sections:
            req_sections.append(rs)
    enforce_ref = spec.get("enforce_referee_sections", True)
    referee_sections_set = set(ref_explicit)
    if enforce_ref:
        for sec in req_sections:
            if REFEREE_SECTION_RE.search(sec):
                referee_sections_set.add(sec)
    referee_sections = [s for s in req_sections if s in referee_sections_set]

    missing_sections = []
    empty_sections = []
    duplicate_sections = []
    stub_sections = []
    thin_sections = []
    section_sizes = {}
    if req_sections:
        positions = []
        for sec in req_sections:
            pat = section_heading_regex(sec)
            matches = list(pat.finditer(text))
            if not matches:
                missing_sections.append(sec)
            else:
                positions.append((matches[0].start(), matches[0].end(), sec))
                if len(matches) > 1:
                    duplicate_sections.append(sec)
        positions.sort()
        sec_min_map = spec.get("section_min_bytes") if isinstance(spec.get("section_min_bytes"), dict) else {}
        base_min = int(spec.get("min_section_bytes") or 0)
        explicit_ref_min = spec.get("referee_min_section_bytes")
        explicit_warn_min = spec.get("warn_section_bytes")
        if explicit_warn_min is not None:
            warn_min = int(explicit_warn_min)
        elif min_bytes >= 4000:
            warn_min = min(800, max(200, min_bytes // max(4, len(req_sections) * 4)))
        else:
            warn_min = 0

        for idx, (_, end_pos, sec) in enumerate(positions):
            next_start = positions[idx + 1][0] if idx + 1 < len(positions) else len(text)
            span = find_section_span(text, sec)
            eff_end = min(span[2], next_start) if span else next_start
            eff_end = max(end_pos, eff_end)
            body = text[end_pos:eff_end].strip()
            body_bytes = len(body.encode("utf-8"))
            section_sizes[sec] = body_bytes
            if len(body) < 20:
                empty_sections.append(sec)
                continue
            if disallow:
                stub_hit = None
                for spat in SECTION_STUB_PATTERNS:
                    sm = re.search(spat, body, re.I)
                    if sm:
                        stub_hit = sm.group(0)
                        break
                if stub_hit:
                    stub_sections.append(sec)
                    issues.append(
                        f"{fname} section {sec} contains unfinished/stub language matching {stub_hit!r}; "
                        f"complete the section before concluding."
                    )
            sec_min = int(sec_min_map.get(sec) or base_min)
            if sec in referee_sections_set:
                if explicit_ref_min is not None:
                    ref_min = int(explicit_ref_min)
                elif min_bytes >= 4000:
                    ref_min = min(1000, max(400, min_bytes // max(2, len(req_sections) * 3)))
                elif min_bytes >= 200:
                    ref_min = max(120, min_bytes // max(2, len(req_sections) * 2))
                else:
                    ref_min = 0
                hard_min = max(sec_min, ref_min)
            else:
                hard_min = sec_min
            if hard_min and body_bytes < hard_min:
                thin_sections.append(sec)
                kind_lbl = "required referee section" if sec in referee_sections_set else "required section"
                issues.append(
                    f"{fname} {kind_lbl} {sec} is too brief ({body_bytes}B < {hard_min}B minimum); "
                    f"complete this section with concrete verified findings rather than leaving a short stub."
                )
            elif warn_min and body_bytes < warn_min and sec not in stub_sections:
                warnings.append(
                    f"{fname} section {sec} looks thin ({body_bytes}B < recommended {warn_min}B); consider expanding it."
                )

    if missing_sections:
        issues.append(f"{fname} is missing required section(s): {', '.join(missing_sections)}.")
    if empty_sections:
        issues.append(f"{fname} has empty/stub section(s): {', '.join(empty_sections)}.")
    if duplicate_sections:
        issues.append(
            f"{fname} contains duplicate heading(s) for required section(s): {', '.join(duplicate_sections)}; "
            f"consolidate or @REVERT."
        )

    section_owners = spec.get("section_owners") if isinstance(spec.get("section_owners"), dict) else {}
    enforce_owners = bool(spec.get("enforce_section_owners", False))
    unauthored_by_owner = {}
    if section_owners:
        for sec_key, expected_owner in section_owners.items():
            owner_str = str(expected_owner or "").strip()
            if not owner_str or sec_key in missing_sections or sec_key in empty_sections:
                continue
            owner_norm = owner_str.upper()
            owner_pat = section_heading_regex(sec_key)
            authored = False
            for entry in manager.log:
                if not entry.get("ok") or entry.get("name") != fname:
                    continue
                if str(entry.get("agent") or "").strip().upper() != owner_norm:
                    continue
                op = str(entry.get("op") or "").lower()
                if op in ("replace", "create"):
                    authored = True
                    break
                if op in ("replace_section", "append_section") and entry.get("section"):
                    if owner_pat.match(str(entry["section"]).strip()):
                        authored = True
                        break
            if not authored:
                unauthored_by_owner[sec_key] = owner_str
                if enforce_owners:
                    issues.append(
                        f"{fname} section {sec_key} has not been authored by designated owner {owner_str} "
                        f"(only peer additions or seed); {owner_str} must write it via @SECTION."
                    )
                else:
                    warnings.append(
                        f"{fname} section {sec_key} has not yet been authored by designated owner {owner_str}."
                    )

    details = {
        "artifact": fname,
        "version": version,
        "size_bytes": size_bytes,
        "min_bytes": min_bytes,
        "required_sections_count": len(req_sections),
        "matched_sections_count": len(req_sections) - len(missing_sections),
        "missing_sections": missing_sections,
        "empty_sections": empty_sections,
        "duplicate_sections": duplicate_sections,
        "stub_sections": stub_sections,
        "thin_sections": thin_sections,
        "referee_sections": referee_sections,
        "section_sizes": section_sizes,
        "unauthored_by_owner": unauthored_by_owner,
        "warnings": warnings,
    }
    return (len(issues) == 0), issues, details
