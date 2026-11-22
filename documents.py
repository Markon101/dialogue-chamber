"""Read-only Reference Document Library & Document Manifest for the Dialogue Chamber.

Allows experiments (and resumed runs) to mount external research papers, specs, or corpora
via a declarative JSON manifest. Agents see a compact index in context and retrieve exact
excerpts on demand via @DOC_READ <doc_id> [offset] and @DOC_SEARCH <doc_id|all> <query>.
"""
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DOC_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,47}$")

_SYMBOL_MAP = {
    "∆": " delta ",
    "Δ": " delta ",
    "θ": " theta ",
    "Θ": " theta ",
    "χ": " chi ",
    "π": " pi ",
    "φ": " phi ",
    "ϕ": " phi ",
    "β": " beta ",
    "λ": " lambda ",
    "Λ": " lambda ",
    "σ": " sigma ",
    "ζ": " zeta ",
    "ε": " epsilon ",
    "ϵ": " epsilon ",
    "−": "-",
    "–": "-",
    "—": "-",
    "⩽": "<=",
    "≤": "<=",
    "⩾": ">=",
    "≥": ">=",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
}

_SEARCH_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "in", "is", "it", "of", "on", "or", "that", "the", "to", "with",
}


class DocumentError(Exception):
    """A reference document operation failed or was refused."""


@dataclass
class DocumentEntry:
    id: str
    title: str
    authors: str = ""
    year: str = ""
    summary: str = ""
    source_url: str = ""
    path: Path = field(default_factory=Path)
    tags: List[str] = field(default_factory=list)
    chars: int = 0
    lines: int = 0
    bytes: int = 0
    sha256: str = ""


def _normalize_with_map(text: str) -> Tuple[str, List[int]]:
    """Normalize text for search while tracking exact raw char offsets.

    Maps common math/Greek symbols and PDF ligatures to ASCII, lowercases text,
    and collapses whitespace runs (including PDF line wraps) to single spaces.
    """
    out_chars: List[str] = []
    idx_map: List[int] = []
    prev_space = True

    for raw_idx, ch in enumerate(text):
        mapped = _SYMBOL_MAP.get(ch)
        if mapped is None:
            mapped = " " if ch.isspace() else ch.lower()
        for mch in mapped:
            if mch.isspace():
                if not prev_space:
                    out_chars.append(" ")
                    idx_map.append(raw_idx)
                    prev_space = True
            else:
                out_chars.append(mch)
                idx_map.append(raw_idx)
                prev_space = False

    if out_chars and out_chars[-1] == " ":
        out_chars.pop()
        idx_map.pop()

    return "".join(out_chars), idx_map


def _normalize_query(raw_query: str) -> str:
    """Clean quotes/carets and normalize symbols/whitespace in a search query."""
    cleaned = raw_query.replace('"', " ").replace("'", " ").replace("`", " ").replace("^", " ")
    norm, _ = _normalize_with_map(cleaned)
    return norm.strip()


class DocumentLibrary:
    """Immutable library of text/markdown reference documents backed by a manifest."""

    def __init__(
        self,
        manifest_path: Optional[Any] = None,
        inline_docs: Optional[List[Dict[str, Any]]] = None,
        base_dir: Optional[Any] = None,
    ):
        self.manifest_path: Optional[Path] = Path(manifest_path).resolve() if manifest_path else None
        self.base_dir: Path = (
            self.manifest_path.parent
            if self.manifest_path is not None
            else (Path(base_dir).resolve() if base_dir else Path.cwd().resolve())
        )
        self.entries: Dict[str, DocumentEntry] = {}
        self._texts: Dict[str, str] = {}
        self._norm_cache: Dict[str, Tuple[str, List[int]]] = {}

        raw_items: List[Tuple[Dict[str, Any], Path]] = []
        if self.manifest_path is not None:
            if not self.manifest_path.exists() or not self.manifest_path.is_file():
                raise DocumentError(f"document manifest not found: {self.manifest_path}")
            try:
                loaded = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise DocumentError(f"failed to parse document manifest {self.manifest_path}: {exc}") from exc
            if isinstance(loaded, dict):
                docs_list = loaded.get("documents", [])
            elif isinstance(loaded, list):
                docs_list = loaded
            else:
                raise DocumentError("document manifest must be a JSON list or an object with 'documents'")
            if not isinstance(docs_list, list):
                raise DocumentError("document manifest 'documents' field must be a list")
            for item in docs_list:
                raw_items.append((item, self.manifest_path.parent))

        if inline_docs:
            anchor = Path(base_dir).resolve() if base_dir else self.base_dir
            for item in inline_docs:
                raw_items.append((item, anchor))

        for item, anchor_dir in raw_items:
            self._register(item, anchor_dir)

    @staticmethod
    def normalize_id(raw_id: Any) -> str:
        s = str(raw_id or "").strip().lower()
        s = re.sub(r"^[<\[(\"'`*@#]+|[>\])\"'`*:.,!]+$", "", s).strip()
        for ext in (".txt", ".md"):
            if s.endswith(ext):
                s = s[: -len(ext)]
        if not DOC_ID_RE.match(s):
            raise DocumentError(
                f"invalid document id {raw_id!r}: use lowercase letters, digits, '-' or '_'"
            )
        return s

    def _register(self, item: Any, anchor_dir: Path) -> None:
        if not isinstance(item, dict):
            raise DocumentError("each document manifest entry must be an object")
        doc_id = self.normalize_id(item.get("id"))
        if doc_id in self.entries:
            raise DocumentError(f"duplicate document id {doc_id!r} in manifest")
        title = str(item.get("title") or doc_id).strip()
        authors = str(item.get("authors") or "").strip()
        year = str(item.get("year") or "").strip()
        summary = str(item.get("summary") or "").strip()
        source_url = str(item.get("source_url") or item.get("url") or "").strip()
        raw_tags = item.get("tags") or []
        tags = [str(t).strip() for t in raw_tags if str(t).strip()] if isinstance(raw_tags, list) else []

        if "text" in item and item["text"] is not None and not item.get("path"):
            text = str(item["text"])
            raw_bytes = text.encode("utf-8")
            resolved = anchor_dir / f"{doc_id}.txt"
        else:
            raw_path = str(item.get("path") or "").strip()
            if not raw_path:
                raise DocumentError(f"document {doc_id!r} missing 'path' (or inline 'text')")
            p = Path(raw_path)
            resolved = (p if p.is_absolute() else (anchor_dir / p)).resolve()
            if not resolved.exists() or not resolved.is_file():
                raise DocumentError(f"document file not found for {doc_id!r}: {resolved}")
            raw_bytes = resolved.read_bytes()
            text = raw_bytes.decode("utf-8", "replace")

        digest = hashlib.sha256(raw_bytes).hexdigest()
        entry = DocumentEntry(
            id=doc_id,
            title=title,
            authors=authors,
            year=year,
            summary=summary,
            source_url=source_url,
            path=resolved,
            tags=tags,
            chars=len(text),
            lines=len(text.splitlines()),
            bytes=len(raw_bytes),
            sha256=digest,
        )
        self.entries[doc_id] = entry
        self._texts[doc_id] = text

    def _normalized(self, doc_id: str) -> Tuple[str, List[int]]:
        cached = self._norm_cache.get(doc_id)
        if cached is None:
            cached = _normalize_with_map(self._texts[doc_id])
            self._norm_cache[doc_id] = cached
        return cached

    def has_docs(self) -> bool:
        return bool(self.entries)

    def ids(self) -> List[str]:
        return list(self.entries.keys())

    def manifest_metadata(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": e.id,
                "title": e.title,
                "authors": e.authors,
                "year": e.year,
                "summary": e.summary,
                "source_url": e.source_url,
                "path": str(e.path),
                "tags": list(e.tags),
                "chars": e.chars,
                "lines": e.lines,
                "bytes": e.bytes,
                "sha256": e.sha256,
            }
            for e in self.entries.values()
        ]

    def render_manifest(self, max_summary_chars: int = 260) -> str:
        if not self.entries:
            return ""
        lines = [
            "--- Reference Document Library (Read-Only Manifest) ---",
            "Use `@DOC_SEARCH <doc_id|all> <keywords>` to find exact character offsets, or `@DOC_READ <doc_id> [offset]` to read full passages.",
        ]
        for e in self.entries.values():
            meta_parts = []
            if e.authors:
                meta_parts.append(e.authors)
            if e.year:
                meta_parts.append(e.year)
            meta_parts.append(f"{e.chars} chars")
            if e.tags:
                meta_parts.append("tags: " + ", ".join(e.tags[:5]))
            summary = e.summary[:max_summary_chars] + ("..." if len(e.summary) > max_summary_chars else "")
            line = f"- `{e.id}`: **{e.title}** ({' | '.join(meta_parts)})"
            if summary:
                line += f" — {summary}"
            lines.append(line)
        return "\n".join(lines)

    def read(self, raw_doc_id: Any, offset: int = 0, max_chars: int = 12000) -> Tuple[str, str, str, int, int]:
        try:
            doc_id = self.normalize_id(raw_doc_id)
        except DocumentError:
            avail = ", ".join(self.ids()) or "none"
            raise DocumentError(f"unknown document {raw_doc_id!r} (available: {avail})")
        if doc_id not in self.entries:
            avail = ", ".join(self.ids()) or "none"
            raise DocumentError(f"no reference document {doc_id!r} (available: {avail})")
        entry = self.entries[doc_id]
        text = self._texts[doc_id]
        if offset < 0 or offset >= max(1, len(text)):
            raise DocumentError(
                f"offset {offset} out of range for {doc_id} ({len(text)} chars); use 0 <= offset < {len(text)}"
            )
        excerpt = text[offset : offset + max_chars]
        return doc_id, entry.title, excerpt, offset, len(text)

    def search(
        self,
        raw_target: Any,
        query: Any,
        max_hits: int = 6,
        window: int = 380,
    ) -> Tuple[str, List[Dict[str, Any]], str]:
        q = str(query or "").strip()
        if not q:
            raise DocumentError("DOC_SEARCH requires a non-empty search query")
        tgt_raw = str(raw_target or "all").strip()
        if not tgt_raw or tgt_raw.lower() in ("all", "*"):
            target_ids = self.ids()
            scope = "all"
            # Handle redundant "all <doc_id> <rest>" patterns
            parts = q.split(None, 1)
            if len(parts) == 2 and parts[1].strip():
                try:
                    maybe_id = self.normalize_id(parts[0])
                except DocumentError:
                    maybe_id = ""
                if maybe_id in self.entries:
                    target_ids = [maybe_id]
                    scope = maybe_id
                    q = parts[1].strip()
        else:
            doc_id = self.normalize_id(tgt_raw)
            if doc_id not in self.entries:
                avail = ", ".join(self.ids()) or "none"
                raise DocumentError(f"no reference document {doc_id!r} (available: {avail})")
            target_ids = [doc_id]
            scope = doc_id

        norm_q = _normalize_query(q)
        if not norm_q:
            raise DocumentError("DOC_SEARCH requires a non-empty search query")

        hits: List[Dict[str, Any]] = []
        seen_ranges: List[Tuple[str, int]] = []

        def _is_near_existing(did: str, raw_off: int) -> bool:
            return any(d == did and abs(o - raw_off) < window // 2 for d, o in seen_ranges)

        # Pass 1: normalized phrase search (handles PDF line wraps, quotes, and symbol normalization)
        for did in target_ids:
            raw_text = self._texts[did]
            norm_text, idx_map = self._normalized(did)
            if not norm_text:
                continue
            pos = 0
            per_doc = 0
            while len(hits) < max_hits and per_doc < 3:
                n_idx = norm_text.find(norm_q, pos)
                if n_idx == -1:
                    break
                raw_match = idx_map[n_idx]
                start = max(0, raw_match - window // 3)
                end = min(len(raw_text), raw_match + len(norm_q) + (2 * window) // 3)
                if not _is_near_existing(did, start):
                    snippet = re.sub(r"\s+", " ", raw_text[start:end]).strip()
                    hits.append({
                        "doc_id": did,
                        "title": self.entries[did].title,
                        "offset": start,
                        "match_offset": raw_match,
                        "match_info": "phrase match",
                        "snippet": snippet,
                    })
                    seen_ranges.append((did, start))
                    per_doc += 1
                pos = n_idx + max(1, len(norm_q))

        # Pass 2: ranked keyword-window scoring fallback
        if len(hits) < max_hits:
            raw_tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", norm_q)
            tokens = [t for t in raw_tokens if t not in _SEARCH_STOPWORDS]
            if not tokens:
                tokens = raw_tokens
            # Deduplicate while preserving order
            unique_tokens: List[str] = []
            for tok in tokens:
                if tok not in unique_tokens:
                    unique_tokens.append(tok)

            if unique_tokens:
                term_variants: List[Tuple[str, List[str]]] = []
                for tok in unique_tokens:
                    variants = [tok]
                    if "-" in tok:
                        no_dash = tok.replace("-", "")
                        spaced = tok.replace("-", " ")
                        variants.extend([no_dash, spaced])
                    term_variants.append((tok, variants))

                candidates: List[Tuple[int, int, str, int, int]] = []
                win_norm = max(320, window + 100)
                step_norm = 220
                for did in target_ids:
                    norm_text, idx_map = self._normalized(did)
                    if not norm_text:
                        continue
                    n_len = len(norm_text)
                    for n_start in range(0, n_len, step_norm):
                        chunk = norm_text[n_start : min(n_len, n_start + win_norm)]
                        distinct = 0
                        total_cnt = 0
                        for _, vars_list in term_variants:
                            best_var_cnt = 0
                            for v in vars_list:
                                c = chunk.count(v)
                                if c > best_var_cnt:
                                    best_var_cnt = c
                            if best_var_cnt > 0:
                                distinct += 1
                                total_cnt += best_var_cnt
                        if distinct > 0:
                            raw_start = idx_map[n_start]
                            candidates.append((distinct, total_cnt, did, raw_start, n_start))

                # Rank by highest distinct matched terms, then total hits, then earlier position
                candidates.sort(key=lambda item: (-item[0], -item[1], item[3]))
                m_terms = len(unique_tokens)
                for distinct, total_cnt, did, raw_start, _ in candidates:
                    if len(hits) >= max_hits:
                        break
                    if _is_near_existing(did, raw_start):
                        continue
                    raw_text = self._texts[did]
                    end = min(len(raw_text), raw_start + window)
                    snippet = re.sub(r"\s+", " ", raw_text[raw_start:end]).strip()
                    hits.append({
                        "doc_id": did,
                        "title": self.entries[did].title,
                        "offset": raw_start,
                        "match_offset": raw_start,
                        "match_info": f"matched {distinct}/{m_terms} terms",
                        "snippet": snippet,
                    })
                    seen_ranges.append((did, raw_start))

        if not hits:
            formatted = f"[Ref Doc Search ({scope}) for {q!r}: 0 matches found. Try broader keywords or @DOC_READ <doc_id> 0.]"
        else:
            lines = [f"[Ref Doc Search ({scope}) for {q!r}: {len(hits)} match(es)]"]
            for h in hits:
                info = f" ({h['match_info']})" if h.get("match_info") else ""
                lines.append(f"- `{h['doc_id']}` @ offset {h['offset']}{info}: ...{h['snippet']}...")
            formatted = "\n".join(lines)
        return scope, hits, formatted
