"""
Disk cache for generated masks.

Mask generation (CLAHE + OTSU + components + bbox split + Gaussian blur)
costs ~10-50 ms per image. Across 11k training images and 60 epochs that's
roughly 30 hours of pure preprocessing. Caching is essentially mandatory.

Cache layout
------------
  <cache_dir>/
    config.json                    # generator config used; cache invalidates
                                   # if config_hash differs
    <image_relpath>.npz            # np.savez_compressed of {'masks': arr}

Key choice
----------
We key by the image's relative path inside the dataset, NOT by hash of the
image bytes. This means: if you replace an image file with a different one
at the same path, you'd get stale cache. For our use case (ReID dataset
images are immutable), this is fine and much faster than hashing each load.

If you change the generator config, the cache is invalidated as a whole
(all .npz files become orphaned but harmless — they just stop being read).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np


def _config_hash(config: Dict) -> str:
    """Deterministic hash of a json-serializable config dict."""
    canonical = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


class MaskCache:
    """
    Simple on-disk mask cache.

    Args:
        cache_dir:        directory to store cache files. Created if absent.
        generator_config: dict of generator-defining settings; used to
                          derive a cache key. Cache is invalidated if this
                          changes between runs.
        readonly:         if True, never write new entries (only read).
    """

    def __init__(
        self,
        cache_dir: str,
        generator_config: Optional[Dict] = None,
        readonly: bool = False,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.readonly = readonly

        config = generator_config or {}
        self._current_hash = _config_hash(config)

        # check existing config.json
        marker = self.cache_dir / "config.json"
        self._cache_valid = True
        if marker.exists():
            try:
                with open(marker, "r") as f:
                    stored = json.load(f)
                stored_hash = _config_hash(stored.get("config", {}))
                if stored_hash != self._current_hash:
                    self._cache_valid = False
            except Exception:
                self._cache_valid = False
        elif not readonly:
            # First-time setup: write the config marker
            with open(marker, "w") as f:
                json.dump({"config": config, "hash": self._current_hash}, f, indent=2)

        if not self._cache_valid:
            print(
                f"[MaskCache] config changed; treating cache at {self.cache_dir} "
                f"as invalid. Existing .npz files will be regenerated as accessed."
            )
            if not readonly:
                # Refresh the marker
                with open(marker, "w") as f:
                    json.dump(
                        {"config": config, "hash": self._current_hash}, f, indent=2,
                    )

    def _path_for(self, image_relpath: str) -> Path:
        # Sanitize: image_relpath might contain subdirs (e.g. "image_train/000001.jpg")
        # We want a unique filename per relpath.
        safe = image_relpath.replace(os.sep, "__").replace("/", "__")
        return self.cache_dir / (safe + ".npz")

    def get(self, image_relpath: str) -> Optional[np.ndarray]:
        """Return cached masks or None if not cached / invalid."""
        if not self._cache_valid:
            return None
        path = self._path_for(image_relpath)
        if not path.exists():
            return None
        try:
            with np.load(path) as data:
                arr = data["masks"]
            return arr.astype(np.float32, copy=False)
        except Exception:
            # corrupted file; ignore
            return None

    def put(self, image_relpath: str, masks: np.ndarray) -> None:
        """Store masks for image_relpath."""
        if self.readonly:
            return
        path = self._path_for(image_relpath)
        try:
            np.savez_compressed(path, masks=masks.astype(np.float32, copy=False))
        except Exception as exc:
            # cache write failure is non-fatal
            print(f"[MaskCache] WARNING: failed to write {path}: {exc}")

    def get_or_compute(
        self,
        image_relpath: str,
        compute_fn,
    ) -> np.ndarray:
        """Return cached masks if present, else compute and cache."""
        cached = self.get(image_relpath)
        if cached is not None:
            return cached
        masks = compute_fn()
        self.put(image_relpath, masks)
        return masks
