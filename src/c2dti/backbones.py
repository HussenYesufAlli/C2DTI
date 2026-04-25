"""Utilities for frozen pretrained backbone embeddings.

This module keeps Phase-1 integration non-breaking:
- If frozen NPZ embeddings are provided, it loads and aligns them.
- If files are missing or incompatible, it falls back to deterministic
  string-hash features so existing pipelines continue to run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from src.c2dti.data_utils import build_string_feature_matrix


def _first_2d_array(npz_file: np.lib.npyio.NpzFile) -> Optional[np.ndarray]:
    """Return the first 2D array from an NPZ archive, if any."""
    for key in npz_file.files:
        arr = np.asarray(npz_file[key])
        if arr.ndim == 2:
            return arr
    return None


def load_frozen_entity_embeddings(
    entities: Sequence[str],
    npz_path: Optional[str],
    default_dim: int = 768,
    embeddings_key: str = "embeddings",
    keys_key: str = "keys",
) -> np.ndarray:
    """Load frozen embeddings for entity strings with safe fallback.

    Supported NPZ formats:
    1) Ordered matrix only:
         embeddings: (n_entities, d)
       where row order matches `entities`.
    2) Keyed table:
         keys:       (n_rows,) entity IDs/strings
         embeddings: (n_rows, d)
       where rows are aligned by value lookup.

    If loading fails, this returns deterministic hash features so the
    pipeline remains fully operational.
    """
    n = len(entities)
    if n == 0:
        return np.zeros((0, default_dim), dtype=np.float32)

    # No external embedding file supplied -> deterministic fallback.
    if not npz_path:
        return build_string_feature_matrix(list(entities), vector_size=default_dim)

    file_path = Path(npz_path)
    if not file_path.exists():
        print(f"[Backbone] Missing NPZ file: {file_path}. Using hash fallback.")
        return build_string_feature_matrix(list(entities), vector_size=default_dim)

    try:
        archive = np.load(str(file_path), allow_pickle=True)
    except Exception as exc:
        print(f"[Backbone] Failed reading NPZ ({file_path}): {exc}. Using hash fallback.")
        return build_string_feature_matrix(list(entities), vector_size=default_dim)

    try:
        if embeddings_key in archive.files:
            emb = np.asarray(archive[embeddings_key], dtype=np.float32)
        else:
            emb = _first_2d_array(archive)
            if emb is None:
                raise ValueError("No 2D embedding matrix found in NPZ")
            emb = emb.astype(np.float32)

        if emb.ndim != 2:
            raise ValueError(f"Embeddings must be 2D, got shape {emb.shape}")

        # Case 1: ordered matrix matches dataset order.
        if emb.shape[0] == n:
            return emb

        # Case 2: keyed rows for lookup.
        if keys_key in archive.files:
            keys = np.asarray(archive[keys_key]).astype(str)
            if len(keys) != emb.shape[0]:
                raise ValueError("keys and embeddings row counts do not match")

            key_to_row = {k: i for i, k in enumerate(keys)}
            out = np.zeros((n, emb.shape[1]), dtype=np.float32)
            missing_idx: list[int] = []

            for i, item in enumerate(entities):
                row = key_to_row.get(str(item))
                if row is None:
                    missing_idx.append(i)
                else:
                    out[i] = emb[row]

            if missing_idx:
                # Fill unknown entities with deterministic features in same dimension.
                fallback = build_string_feature_matrix(list(entities), vector_size=emb.shape[1])
                out[missing_idx] = fallback[missing_idx]
                print(
                    f"[Backbone] {len(missing_idx)} / {n} entities missing in {file_path.name}; "
                    "filled with hash fallback."
                )

            return out

        # If row count mismatches and no keys provided, we cannot align safely.
        raise ValueError(
            f"Embedding rows ({emb.shape[0]}) do not match entity count ({n}) and no '{keys_key}' present"
        )
    except Exception as exc:
        print(f"[Backbone] Incompatible NPZ ({file_path}): {exc}. Using hash fallback.")
        return build_string_feature_matrix(list(entities), vector_size=default_dim)


# ---------------------------------------------------------------------------
# MASHead — Pillar 3: Masked AutoEncoder Self-Supervision Head
# ---------------------------------------------------------------------------

class MASHead:
    """Numpy-based Masked AutoEncoder head for Pillar 3 of C2DTI.

    How it works (beginner-friendly explanation):
      Imagine you have 768-dimensional drug embeddings.
      We randomly "cover up" (mask) 15% of those dimensions — say dims 5, 42, 100, ...
      Then we try to *reconstruct* those hidden dims from the other 85% of dims using
      a linear decoder (just a matrix multiplication).
      The reconstruction error (MSE) tells us:
        - Low error  → the masked dims are predictable from the rest → the embeddings
                       carry structured, redundant information (good representations).
        - High error → the masked dims are independent from the rest → sparse/noisy
                       embeddings that are harder to compress.

    This is a global column-mask approach:
      - A fixed set of columns is masked (same for all samples).
      - A single least-squares linear decoder W: R^{d_unmasked} → R^{d_masked} is fitted.
      - No PyTorch or GPU required — pure numpy, fits the existing numpy pipeline.

    Args:
        mask_ratio: Fraction of embedding dimensions to mask (default 0.15 = 15%).
        seed: Random seed for reproducible mask selection.
    """

    def __init__(self, mask_ratio: float = 0.15, seed: int = 42) -> None:
        self.mask_ratio = mask_ratio
        self.seed = seed
        # These are set during fit():
        self._mask: Optional[np.ndarray] = None   # bool array (d,) — True = masked dim
        self._W: Optional[np.ndarray] = None       # (d_unmasked, d_masked) decoder weights
        self._b: Optional[np.ndarray] = None       # (d_masked,) decoder bias

    def _build_mask(self, d: int) -> np.ndarray:
        """Choose which columns to mask — deterministic given seed."""
        rng = np.random.default_rng(self.seed)
        n_masked = max(1, int(d * self.mask_ratio))
        mask = np.zeros(d, dtype=bool)
        mask[rng.choice(d, n_masked, replace=False)] = True
        return mask

    def fit(self, embeddings: np.ndarray) -> "MASHead":
        """Fit the linear decoder on a set of embeddings.

        We solve: X_unmasked @ W ≈ X_masked  (least-squares).

        Args:
            embeddings: (n_samples, d) embedding matrix.

        Returns:
            self — so you can chain: head.fit(e).reconstruct_loss(e).
        """
        if embeddings.ndim != 2:
            raise ValueError(f"MASHead.fit expects 2D input, got shape {embeddings.shape}")

        n, d = embeddings.shape
        self._mask = self._build_mask(d)

        # Split columns into unmasked (inputs) and masked (reconstruction targets).
        X = embeddings[:, ~self._mask].astype(np.float64)   # (n, d_unmasked)
        Y = embeddings[:, self._mask].astype(np.float64)    # (n, d_masked)

        # Add a bias column to X so the decoder can shift the output.
        X_bias = np.hstack([X, np.ones((n, 1), dtype=np.float64)])

        # Least-squares: minimise ||X_bias @ solution - Y||_F^2
        solution, _, _, _ = np.linalg.lstsq(X_bias, Y, rcond=None)
        self._W = solution[:-1]   # (d_unmasked, d_masked)
        self._b = solution[-1]    # (d_masked,)

        return self

    def reconstruct_loss(self, embeddings: np.ndarray) -> float:
        """Compute reconstruction MSE on masked dimensions.

        Args:
            embeddings: (n_samples, d) embedding matrix (can be same or different from fit).

        Returns:
            Mean squared error between predicted and true values for the masked dims.
        """
        if self._mask is None or self._W is None:
            raise RuntimeError("Call MASHead.fit() before reconstruct_loss()")

        X = embeddings[:, ~self._mask].astype(np.float64)   # (n, d_unmasked)
        Y_true = embeddings[:, self._mask].astype(np.float64)  # (n, d_masked)
        Y_pred = X @ self._W + self._b                          # (n, d_masked)

        return float(np.mean((Y_pred - Y_true) ** 2))


# ---------------------------------------------------------------------------
# SequenceViewEncoder — Pillar 1: Dual Backbone Sequence View
# ---------------------------------------------------------------------------

class SequenceViewEncoder:
    """Character n-gram encoder for drug SMILES and protein amino-acid sequences.

    Beginner-friendly explanation:
    --------------------------------
    A SMILES string like "CC(=O)Nc1ccc(O)cc1" is just text.
    We turn it into a fixed-size number vector by counting how often each short
    sub-string (n-gram) appears.  For example with n=2:
      "CCN" → bigrams: "CC", "CN"  → count each one.
    We hash every n-gram into a bucket (0 … vocab_size-1) using Python's built-in
    hash function, then count how many n-grams fell into each bucket.
    That count vector is the embedding for this molecule.

    The same idea works for proteins: "MKTAYIAKQR…" counted with n=3 (trigrams).

    Why this is useful for C2DTI Pillar 1:
    - Requires NO pre-training, NO GPU, NO external files.
    - Is deterministic — same string → same vector, every run.
    - Provides a "raw sequence view" independent of the frozen NPZ embeddings,
      which is exactly what the causal dual-backbone (Pillar 1) needs.

    Args:
        ngram_n:    Length of character n-grams (default 2 for drugs, 3 for proteins).
        vocab_size: Number of hash buckets = output embedding dimension.
        normalize:  If True, L2-normalize each row so magnitudes are comparable.
    """

    def __init__(
        self,
        ngram_n: int = 2,
        vocab_size: int = 512,
        normalize: bool = True,
    ) -> None:
        self.ngram_n = ngram_n
        self.vocab_size = vocab_size
        self.normalize = normalize

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ngrams(self, text: str) -> list:
        """Extract all overlapping n-grams from a string.

        Pads strings shorter than ngram_n so every input produces at least
        one n-gram and the encoder never silently returns an all-zero vector.
        Example: _ngrams("ACGT", n=2) → ["AC", "CG", "GT"]
        """
        n = self.ngram_n
        if len(text) < n:
            text = text.ljust(n, "_")
        return [text[i : i + n] for i in range(len(text) - n + 1)]

    def _encode_one(self, text: str) -> np.ndarray:
        """Encode a single string into a vocab_size-dimensional count vector.

        Steps:
        1. Extract n-grams.
        2. Hash each n-gram into a bucket index (0 … vocab_size-1).
        3. Count hits per bucket.
        4. Optionally L2-normalize.
        """
        vec = np.zeros(self.vocab_size, dtype=np.float32)
        for gram in self._ngrams(text):
            # Python hash can be negative → abs + mod maps into [0, vocab_size).
            bucket = abs(hash(gram)) % self.vocab_size
            vec[bucket] += 1.0
        if self.normalize:
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
        return vec

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, sequences: Sequence[str]) -> np.ndarray:
        """Encode a list of strings into an (n_sequences, vocab_size) matrix.

        Args:
            sequences: List of SMILES or amino-acid strings.

        Returns:
            Float32 array of shape (len(sequences), vocab_size).
        """
        if len(sequences) == 0:
            return np.zeros((0, self.vocab_size), dtype=np.float32)
        return np.stack([self._encode_one(s) for s in sequences], axis=0)

    def __repr__(self) -> str:
        return (
            f"SequenceViewEncoder(ngram_n={self.ngram_n}, "
            f"vocab_size={self.vocab_size}, normalize={self.normalize})"
        )
