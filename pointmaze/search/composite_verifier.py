import torch


class CompositeVerifier:
    """
    Combines multiple verifiers by summing their weighted logprobs/gradients.

    Provides the same interface as MazeVerifier and DistanceFieldVerifier so
    it can be used as a drop-in replacement for either.

    Args:
        verifiers: list of verifier objects (MazeVerifier, DistanceFieldVerifier, …)
        weights:   list of float weights, one per verifier
    """

    def __init__(self, verifiers: list, weights: list):
        assert len(verifiers) == len(weights), "verifiers and weights must have equal length"
        self.verifiers = verifiers
        self.weights   = weights
        self.last_stats = {}

    def update_env(self, env, **kwargs):
        for v in self.verifiers:
            v.update_env(env)

    def get_guidance(self, x, return_logp=False, **kwargs):
        """
        Sum weighted contributions from all component verifiers.

        Signature matches MazeVerifier.get_guidance (accepts the same **kwargs).
        """
        if return_logp:
            total = None
            stats = {}
            for v, w in zip(self.verifiers, self.weights):
                lp = v.get_guidance(x, return_logp=True, **kwargs)
                name = v.__class__.__name__
                if hasattr(v, "last_stats"):
                    stats[name] = dict(v.last_stats)
                    stats[name]["composite_weight"] = float(w)
                # expose each verifier's weighted contribution so callers can
                # separate hard violations (MazeVerifier) from soft preferences
                # (DistanceFieldVerifier) -- e.g. DFS thresholds on the former
                # while still ranking by the combined total.
                stats.setdefault(name, {})
                stats[name]["weighted_logp_sum"] = float((w * lp).sum().item())
                total = w * lp if total is None else total + w * lp
            self.last_stats = stats
            if total is None:
                return torch.zeros(x.shape[0], device=x.device)
            return total
        else:
            total = None
            for v, w in zip(self.verifiers, self.weights):
                g = v.get_guidance(x, return_logp=False, **kwargs)
                total = w * g if total is None else total + w * g
            if total is None:
                return torch.zeros_like(x)
            return total
