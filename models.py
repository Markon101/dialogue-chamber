"""Dynamic OpenRouter Model Discovery, Catalog Caching, Filtering, and Free-Tier Management.

Provides live model discovery from OpenRouter API, local caching with TTL,
rich search/filtering (free-tier, context length, pricing, reasoning), and safe cost gating.
"""
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

CACHE_DIR = Path(__file__).resolve().parent / ".cache"
DEFAULT_CACHE_FILE = CACHE_DIR / "openrouter_models.json"
CACHE_TTL_SECONDS = 86400  # 24 hours


@dataclass
class ModelMetadata:
    """Standardized metadata for a discovered inference model."""
    id: str
    name: str
    backend: str = "openrouter"
    context_length: int = 4096
    max_completion_tokens: Optional[int] = None
    prompt_price_per_m: float = 0.0      # USD per 1M prompt tokens
    completion_price_per_m: float = 0.0  # USD per 1M completion tokens
    is_free: bool = False
    has_reasoning: bool = False
    description: str = ""
    modality: str = "text->text"
    raw_data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("raw_data", None)
        return d


class OpenRouterCatalog:
    """Manages OpenRouter model discovery, caching, searching, and pricing inspection."""

    def __init__(self, cache_file: Optional[Path] = None, ttl_s: int = CACHE_TTL_SECONDS):
        self.cache_file = cache_file or DEFAULT_CACHE_FILE
        self.ttl_s = ttl_s
        self._models: Dict[str, ModelMetadata] = {}
        self._last_fetched: float = 0.0
        self._load_cache()

    def _load_cache(self) -> bool:
        if not self.cache_file.exists():
            return False
        try:
            data = json.loads(self.cache_file.read_text(encoding="utf-8"))
            self._last_fetched = float(data.get("fetched_at", 0))
            models_raw = data.get("models", [])
            self._models.clear()
            for m in models_raw:
                meta = ModelMetadata(
                    id=m["id"],
                    name=m.get("name", m["id"]),
                    backend="openrouter",
                    context_length=int(m.get("context_length", 4096)),
                    max_completion_tokens=m.get("max_completion_tokens"),
                    prompt_price_per_m=float(m.get("prompt_price_per_m", 0.0)),
                    completion_price_per_m=float(m.get("completion_price_per_m", 0.0)),
                    is_free=bool(m.get("is_free", False)),
                    has_reasoning=bool(m.get("has_reasoning", False)),
                    description=m.get("description", ""),
                    modality=m.get("modality", "text->text"),
                )
                self._models[meta.id] = meta
            return len(self._models) > 0
        except Exception:
            return False

    def _save_cache(self, raw_list: List[Dict[str, Any]]) -> None:
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            export_data = {
                "fetched_at": self._last_fetched,
                "count": len(self._models),
                "models": [m.to_dict() for m in self._models.values()],
            }
            self.cache_file.write_text(json.dumps(export_data, indent=2), encoding="utf-8")
        except Exception:
            pass

    def refresh(self, force: bool = False, timeout_s: float = 8.0) -> bool:
        """Fetches live model listing from OpenRouter API."""
        now = time.time()
        if not force and (now - self._last_fetched) < self.ttl_s and len(self._models) > 0:
            return True

        url = "https://openrouter.ai/api/v1/models"
        headers = {"User-Agent": "DialogueChamber/2.0"}
        # Include API key if available in env, though not strictly required for listing
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                if resp.status != 200:
                    return False
                payload = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            # If network fails, return cached if available
            if len(self._models) > 0:
                return True
            return False

        data_list = payload.get("data", [])
        if not isinstance(data_list, list):
            return False

        self._models.clear()
        self._last_fetched = now

        for item in data_list:
            mid = item.get("id", "")
            if not mid:
                continue

            pricing = item.get("pricing") or {}
            try:
                p_prompt = float(pricing.get("prompt", 0.0)) * 1_000_000.0
            except (ValueError, TypeError):
                p_prompt = 0.0

            try:
                p_comp = float(pricing.get("completion", 0.0)) * 1_000_000.0
            except (ValueError, TypeError):
                p_comp = 0.0

            # Free status: zero cost or :free suffix
            is_free = (p_prompt == 0.0 and p_comp == 0.0) or mid.endswith(":free")

            top_p = item.get("top_provider") or {}
            max_comp = top_p.get("max_completion_tokens")

            desc = item.get("description", "")
            arch = item.get("architecture") or {}
            inst_type = str(arch.get("instruct_type") or "").lower()
            has_reasoning = (
                "reason" in desc.lower()
                or "thinking" in desc.lower()
                or "deepseek-r1" in mid.lower()
                or "reasoning" in inst_type
            )

            meta = ModelMetadata(
                id=mid,
                name=item.get("name", mid),
                backend="openrouter",
                context_length=int(item.get("context_length", 4096)),
                max_completion_tokens=int(max_comp) if max_comp else None,
                prompt_price_per_m=round(p_prompt, 4),
                completion_price_per_m=round(p_comp, 4),
                is_free=is_free,
                has_reasoning=has_reasoning,
                description=desc[:240],
                modality=arch.get("modality", "text->text"),
                raw_data=item,
            )
            self._models[meta.id] = meta

        self._save_cache(data_list)
        return True

    def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        """Gets metadata for model_id, attempting refresh if missing."""
        if not self._models:
            self.refresh(force=False)
        return self._models.get(model_id)

    def list_models(
        self,
        free_only: bool = False,
        provider: Optional[str] = None,
        min_ctx: Optional[int] = None,
        reasoning: Optional[bool] = None,
        limit: int = 50,
    ) -> List[ModelMetadata]:
        """Lists models matching criteria."""
        if not self._models:
            self.refresh(force=False)

        results = []
        for m in self._models.values():
            if free_only and not m.is_free:
                continue
            if provider:
                # check provider prefix, e.g. "google/", "meta-llama/", "qwen/"
                prov_part = m.id.split("/")[0] if "/" in m.id else ""
                if provider.lower() not in prov_part.lower() and provider.lower() not in m.name.lower():
                    continue
            if min_ctx and m.context_length < min_ctx:
                continue
            if reasoning is not None and m.has_reasoning != reasoning:
                continue
            results.append(m)

        # Sort: free models first, then by context length descending
        results.sort(key=lambda x: (not x.is_free, -x.context_length, x.id))
        return results[:limit]

    def search_models(self, query: str, free_only: bool = False, limit: int = 30) -> List[ModelMetadata]:
        """Searches model ID, name, or description."""
        if not self._models:
            self.refresh(force=False)

        q = query.strip().lower()
        results = []
        for m in self._models.values():
            if free_only and not m.is_free:
                continue
            if q in m.id.lower() or q in m.name.lower() or q in m.description.lower():
                results.append(m)

        results.sort(key=lambda x: (not x.is_free, -x.context_length, x.id))
        return results[:limit]


# Global singleton catalog instance
_CATALOG_INSTANCE: Optional[OpenRouterCatalog] = None


def get_catalog() -> OpenRouterCatalog:
    global _CATALOG_INSTANCE
    if _CATALOG_INSTANCE is None:
        _CATALOG_INSTANCE = OpenRouterCatalog()
    return _CATALOG_INSTANCE


def format_models_table(models: List[ModelMetadata]) -> str:
    """Renders human-readable CLI table of models."""
    if not models:
        return "No models matched the given criteria."

    lines = [
        f"{'MODEL IDENTIFIER':<46} {'FREE':<5} {'CONTEXT':<8} {'PROMPT/1M':<10} {'COMPL/1M':<10} {'THINK':<5}",
        "-" * 88,
    ]
    for m in models:
        free_str = "YES" if m.is_free else "NO"
        think_str = "YES" if m.has_reasoning else "NO"
        ctx_str = f"{m.context_length // 1024}K" if m.context_length >= 1024 else f"{m.context_length}"
        p_str = f"${m.prompt_price_per_m:.2f}" if m.prompt_price_per_m > 0 else "FREE"
        c_str = f"${m.completion_price_per_m:.2f}" if m.completion_price_per_m > 0 else "FREE"
        lines.append(
            f"{m.id:<46} {free_str:<5} {ctx_str:<8} {p_str:<10} {c_str:<10} {think_str:<5}"
        )
    return "\n".join(lines)


def handle_models_cli(argv: List[str]) -> int:
    """CLI subcommand dispatcher for 'models list', 'models search', 'models info', 'models refresh'."""
    cat = get_catalog()
    if not argv or argv[0] in ("-h", "--help", "help"):
        print("Dialogue Chamber — OpenRouter Model Catalog Commands:\n")
        print("  list [--free] [--provider PROV] [--min-ctx N] [--limit N]   List models")
        print("  search <query> [--free] [--limit N]                        Search models")
        print("  info <model_id>                                            Detailed model info")
        print("  refresh                                                    Force refresh API cache")
        return 0

    sub = argv[0].lower()
    if sub == "refresh":
        print("[Catalog] Refreshing live OpenRouter model catalog...")
        ok = cat.refresh(force=True)
        if ok:
            print(f"[Catalog] Success! Cached {len(cat._models)} models in {cat.cache_file}")
            return 0
        else:
            print("[Catalog] Warning: Failed to refresh from live API; using existing cache if present.")
            return 1

    elif sub in ("list", "ls"):
        free_only = "--free" in argv
        provider = None
        min_ctx = None
        limit = 40
        for i, a in enumerate(argv[1:], 1):
            if a in ("--provider", "-p") and i + 1 < len(argv):
                provider = argv[i + 1]
            elif a in ("--min-ctx", "-c") and i + 1 < len(argv):
                try:
                    min_ctx = int(argv[i + 1])
                except ValueError:
                    pass
            elif a in ("--limit", "-n") and i + 1 < len(argv):
                try:
                    limit = int(argv[i + 1])
                except ValueError:
                    pass

        models = cat.list_models(free_only=free_only, provider=provider, min_ctx=min_ctx, limit=limit)
        print(f"\nOPENROUTER MODELS (Showing {len(models)} models, free_only={free_only}):\n")
        print(format_models_table(models))
        print(f"\nCatalog cache: {cat.cache_file} (cached {len(cat._models)} models)")
        return 0

    elif sub in ("search", "find"):
        if len(argv) < 2:
            print("Usage: python3 chamber.py models search <query> [--free]")
            return 1
        query = argv[1]
        free_only = "--free" in argv
        results = cat.search_models(query, free_only=free_only)
        print(f"\nSEARCH RESULTS FOR '{query}' (Found {len(results)} matches):\n")
        print(format_models_table(results))
        return 0

    elif sub == "info":
        if len(argv) < 2:
            print("Usage: python3 chamber.py models info <model_id>")
            return 1
        mid = argv[1]
        m = cat.get_model(mid)
        if not m:
            print(f"Error: Model '{mid}' not found in catalog. Try: python3 chamber.py models refresh")
            return 1
        print(f"\nMODEL INFORMATION: {m.id}")
        print(f"  Name:               {m.name}")
        print(f"  Context Window:     {m.context_length} tokens ({m.context_length // 1024}K)")
        print(f"  Max Output Tokens:  {m.max_completion_tokens or 'Provider default'}")
        print(f"  Free Tier:          {'YES' if m.is_free else 'NO'}")
        print(f"  Prompt Pricing:     ${m.prompt_price_per_m:.4f} / 1M tokens")
        print(f"  Completion Pricing: ${m.completion_price_per_m:.4f} / 1M tokens")
        print(f"  Reasoning Support:  {'YES' if m.has_reasoning else 'NO / Unreported'}")
        print(f"  Modality:           {m.modality}")
        if m.description:
            print(f"  Description:        {m.description}")
        return 0

    else:
        print(f"Unknown subcommand: {sub}. Use: list, search, info, refresh.")
        return 1
