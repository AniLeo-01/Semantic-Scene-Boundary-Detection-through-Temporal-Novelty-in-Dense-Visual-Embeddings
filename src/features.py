"""DINOv3 feature extraction. Returns L2-normalized CLS + patch-mean embeddings.

Falls back to DINOv2 if DINOv3 weights are unavailable (HF gating, offline, etc).

Authentication for gated repos: set ``HF_TOKEN`` (or the HuggingFace-standard
``HUGGING_FACE_HUB_TOKEN``) in the environment before importing. DINOv3 is
gated on HuggingFace; without a token the loader silently falls through to
DINOv2.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


def _hf_token() -> Optional[str]:
    """Read the HuggingFace token from env. Empty strings count as missing."""
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        tok = os.environ.get(var)
        if tok:
            return tok
    return None


DEFAULT_MODELS = [
    # Empirically tied with DINOv3 on Charades and not gated — see THESIS.md.
    "facebook/dinov2-small",
    "facebook/dinov2-base",
    "facebook/dinov3-vits16-pretrain-lvd1689m",  # only useful if you have HF access
]


@dataclass
class FrameEmbedding:
    idx: int
    pts_s: float
    cls: np.ndarray            # (D,)   L2-normalized
    patch_mean: np.ndarray     # (D,)   L2-normalized mean of patch tokens
    combined: np.ndarray       # (2D,)  concat of [cls, patch_mean] then L2-normalized
    patches: np.ndarray        # (N, D) raw patch tokens, L2-normalized per row


class DinoFeatureExtractor:
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
        token: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        # Explicit > env > None. Env is read once at construction.
        self.token = token if token is not None else _hf_token()

        import sys as _sys
        candidates = [model_name] if model_name else DEFAULT_MODELS
        explicit = model_name is not None
        last_err = None
        for name in candidates:
            print(f"[features] trying {name}", file=_sys.stderr, flush=True)
            try:
                self.processor = AutoImageProcessor.from_pretrained(name, token=self.token)
                self.model = (
                    AutoModel.from_pretrained(name, torch_dtype=dtype, token=self.token)
                    .to(self.device)
                    .eval()
                )
                self.model_name = name
                print(f"[features] loaded {name}", file=_sys.stderr, flush=True)
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                print(f"[features] FAILED {name}: {type(e).__name__}: {e}",
                      file=_sys.stderr, flush=True)
                # If the caller named a model explicitly, don't silently fall
                # through to a different one — surface the real error.
                if explicit:
                    raise
                continue
        else:
            extra = ""
            if self.token is None:
                extra = (
                    "\nNote: no HuggingFace token detected. DINOv3 is gated — "
                    "set HF_TOKEN in the env, or pass `token=...` to "
                    "DinoFeatureExtractor."
                )
            raise RuntimeError(f"Could not load any DINO model. Last error: {last_err}{extra}")

    @torch.inference_mode()
    def embed_batch(self, images: List[np.ndarray], idxs: List[int], pts: List[float]) -> List[FrameEmbedding]:
        """Embed a batch of RGB uint8 HxWx3 arrays."""
        pil = [Image.fromarray(img) for img in images]
        inputs = self.processor(images=pil, return_tensors="pt").to(self.device)
        # Some DINO processors don't accept dtype; cast manually.
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)

        out = self.model(**inputs)
        # DINOv2/v3 expose last_hidden_state: (B, 1+N, D) where token 0 is CLS.
        hs = out.last_hidden_state          # (B, 1+N, D)
        cls = hs[:, 0, :]                   # (B, D)
        patches = hs[:, 1:, :]              # (B, N, D)
        patch_mean = patches.mean(dim=1)    # (B, D)

        cls = F.normalize(cls, dim=-1)
        patch_mean = F.normalize(patch_mean, dim=-1)
        # L2-normalize each patch token row-wise so per-patch cosine sim
        # downstream is just a dot product.
        patches_n = F.normalize(patches, dim=-1)
        combined = F.normalize(torch.cat([cls, patch_mean], dim=-1), dim=-1)

        cls_np = cls.float().cpu().numpy()
        pm_np = patch_mean.float().cpu().numpy()
        cb_np = combined.float().cpu().numpy()
        pt_np = patches_n.float().cpu().numpy()  # (B, N, D)

        return [
            FrameEmbedding(
                idx=i, pts_s=p,
                cls=cls_np[k],
                patch_mean=pm_np[k],
                combined=cb_np[k],
                patches=pt_np[k],
            )
            for k, (i, p) in enumerate(zip(idxs, pts))
        ]
