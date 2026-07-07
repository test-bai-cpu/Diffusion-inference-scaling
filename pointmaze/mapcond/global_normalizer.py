"""
A single LimitsNormalizer shared across all training maps.

The prior work fits one normalizer per map, so a coordinate normalizes
differently depending on which map produced it. For a map-conditional model
that must be undone: the same world (x, y) has to map to the same normalized
value in every map, otherwise the map grid and the trajectory speak different
coordinate systems and the encoder cannot ground walls against positions.

We therefore fit the repo's own LimitsNormalizer on the *union* of per-key
(observations, actions) ranges across the training maps. Reusing the repo class
(not a re-implementation) guarantees the normalize/unnormalize math is
bit-identical to what the diffusion policy applies at inference time.
"""
import os
import pickle
import numpy as np

from diffuser.datasets.normalization import LimitsNormalizer
from . import data_utils as DU


class GlobalNormalizer:
    """
    Holds one LimitsNormalizer per key, each fitted to the union min/max across
    maps. API mirrors DatasetNormalizer: __call__(x, key) / normalize / unnormalize.
    """

    def __init__(self, bounds):
        # bounds: {key: (min_vec, max_vec)}
        self.bounds = {k: (np.asarray(lo, np.float32), np.asarray(hi, np.float32))
                       for k, (lo, hi) in bounds.items()}
        self.normalizers = {}
        for key, (lo, hi) in self.bounds.items():
            # LimitsNormalizer(X) sets mins=X.min(0), maxs=X.max(0); feed the
            # two extreme rows so it stores exactly [lo, hi].
            X = np.stack([lo, hi], axis=0).astype(np.float32)
            self.normalizers[key] = LimitsNormalizer(X)

    # DatasetNormalizer-compatible surface -----------------------------------
    def __call__(self, x, key):
        return self.normalize(x, key)

    def normalize(self, x, key):
        return self.normalizers[key].normalize(x)

    def unnormalize(self, x, key):
        return self.normalizers[key].unnormalize(x)

    def __repr__(self):
        s = "GlobalNormalizer(\n"
        for k, (lo, hi) in self.bounds.items():
            s += f"  {k}: min={np.round(lo,3).tolist()} max={np.round(hi,3).tolist()}\n"
        return s + ")"

    @property
    def observation_dim(self):
        """DatasetNormalizer-compatible observation dimensionality."""
        return int(self.bounds["observations"][0].shape[-1])

    @property
    def action_dim(self):
        """DatasetNormalizer-compatible action dimensionality."""
        return int(self.bounds["actions"][0].shape[-1])

    # (de)serialization ------------------------------------------------------
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"bounds": self.bounds}, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        return cls(d["bounds"])


def fit_global_normalizer(data_sources):
    """
    Fit a GlobalNormalizer over the union of coordinate ranges across sources.

    `data_sources` may be the legacy list of maze names or explicit map specs
    that point at base-map directories and/or giant variant .npz files.
    """
    bounds = DU.coordinate_bounds(data_sources)
    return GlobalNormalizer(bounds)
