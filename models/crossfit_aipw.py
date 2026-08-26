"""Trajectory-level cross-fitted conditional AIPW estimator.

The estimator is intentionally offline. It never replaces the lower-variance
plug-in D used by the online control path. Nuisance response models are fitted
out of fold at the episode level, row-level AIPW pseudo-outcomes are formed
only on held-out episodes, and a second-stage ridge model estimates E[phi|X].
"""

from dataclasses import dataclass

import numpy as np


def _partition_independent_context_summary(sample):
    """Summarise raw leave-one-target context without learned state.

    Cross-fitted AIPW is part of Paper A's measurement subsystem, so its X
    representation must not consume core latents, peripheral memories, or
    structural beliefs created downstream by the same causal estimates.
    Replay may store either an explicit leave-one-target ``context_items``
    table or one shared ``context_block`` plus the target position.
    """
    items = None
    mask = None

    if sample.get("context_items") is not None:
        items = np.asarray(sample["context_items"], dtype=np.float64)
        if items.ndim != 2:
            raise ValueError("context_items must be a rank-2 raw item table")
        mask = np.asarray(
            sample.get("context_mask", np.ones(items.shape[0])), dtype=bool
        ).reshape(-1)
        if mask.shape != (items.shape[0],):
            raise ValueError("context_mask must have one entry per raw context item")
    elif sample.get("context_block") is not None:
        block = sample["context_block"]
        items = np.asarray(block.get("items"), dtype=np.float64)
        ids = np.asarray(block.get("neighbor_ids"), dtype=np.int64).reshape(-1)
        if items.ndim != 2 or items.shape[0] != ids.shape[0]:
            raise ValueError("context_block raw items/neighbor_ids are malformed")
        position = sample.get("target_context_position")
        if position is None:
            target_id = sample.get("neighbor_id")
            matches = np.flatnonzero(ids == int(target_id)) if target_id is not None else np.asarray([], dtype=np.int64)
            if matches.size != 1:
                raise ValueError(
                    "context_block requires an unambiguous target position for leave-one-out"
                )
            position = int(matches[0])
        position = int(position)
        if position < 0 or position >= items.shape[0]:
            raise ValueError("target_context_position lies outside context_block")
        mask = np.ones(items.shape[0], dtype=bool)
        mask[position] = False
    else:
        raise ValueError(
            "cross-fitted AIPW requires raw partition-independent context_items "
            "or context_block; downstream z/m/belief fallbacks are forbidden"
        )

    if not np.all(np.isfinite(items)):
        raise ValueError("raw context contains NaN or infinity")
    active = items[mask]
    width = int(items.shape[1])
    if active.size == 0:
        summed = np.zeros(width, dtype=np.float64)
        squared = np.zeros(width, dtype=np.float64)
        count = 0.0
    else:
        summed = np.sum(active, axis=0)
        squared = np.sum(np.square(active), axis=0)
        count = float(active.shape[0])
    return np.concatenate([summed, squared, np.asarray([count])])


def proxy_context_features(sample, action_dim):
    """Return a fixed, pre-treatment, partition-independent X vector."""
    raw_action = sample["action_i"]
    if not np.isfinite(raw_action) or int(raw_action) != raw_action:
        raise ValueError("action_i must be a finite integer action identity")
    raw_action = int(raw_action)
    if raw_action < 0 or raw_action >= int(action_dim):
        raise ValueError(
            f"action_i must lie in [0, {int(action_dim)}), got {raw_action}"
        )
    action_i = np.zeros(int(action_dim), dtype=np.float64)
    action_i[raw_action] = 1.0
    fields = (
        np.asarray(sample["obs_i"], dtype=np.float64).reshape(-1),
        action_i,
        np.asarray(sample["pair_feat"], dtype=np.float64).reshape(-1),
        _partition_independent_context_summary(sample),
    )
    out = np.concatenate(fields)
    if not np.all(np.isfinite(out)):
        raise ValueError("cross-fit context contains NaN or infinity")
    return out


@dataclass
class _Ridge:
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray

    def predict(self, x):
        x = np.asarray(x, dtype=np.float64)
        design = np.concatenate(
            [np.ones((x.shape[0], 1)), (x - self.mean) / self.scale], axis=1
        )
        return design @ self.coef


def _fit_ridge(x, y, ridge):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    design = np.concatenate(
        [np.ones((x.shape[0], 1)), (x - mean) / scale], axis=1
    )
    gram = design.T @ design
    penalty = np.eye(gram.shape[0], dtype=np.float64) * float(ridge)
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(gram + penalty, design.T @ y)
    return _Ridge(mean=mean, scale=scale, coef=coef)


class CrossFittedConditionalAIPW:
    """Cross-fit Q by episode, then regress held-out AIPW scores on X."""

    def __init__(self, action_dim, n_folds=5, ridge=1e-3, iw_clip=None):
        self.action_dim = int(action_dim)
        self.n_folds = int(max(2, n_folds))
        self.ridge = float(ridge)
        self.iw_clip = None if iw_clip is None else float(iw_clip)
        self.effect_model = None
        self.diagnostics = {}

    def fit(self, samples):
        rows = [
            sample for sample in samples
            if sample.get("episode_id") is not None
            and sample.get("policy_probs_j") is not None
            and sample.get("behaviour_prob_j") is not None
            and bool(sample.get("horizon_complete", True))
        ]
        episodes = sorted({sample["episode_id"] for sample in rows})
        if len(episodes) < 2:
            raise ValueError("Cross-fitting requires at least two episodes")
        n_folds = min(self.n_folds, len(episodes))
        fold_by_episode = {
            episode: index % n_folds for index, episode in enumerate(episodes)
        }
        x_all = np.stack(
            [proxy_context_features(sample, self.action_dim) for sample in rows]
        )
        phi = np.full(len(rows), np.nan, dtype=np.float64)
        leakage_checks = []

        for fold in range(n_folds):
            train_idx = np.asarray(
                [fold_by_episode[sample["episode_id"]] != fold for sample in rows]
            )
            test_idx = ~train_idx
            train_episodes = {rows[i]["episode_id"] for i in np.flatnonzero(train_idx)}
            test_episodes = {rows[i]["episode_id"] for i in np.flatnonzero(test_idx)}
            leakage_checks.append(not bool(train_episodes & test_episodes))
            pooled_y = np.asarray(
                [rows[i]["target_return_h"] for i in np.flatnonzero(train_idx)]
            )
            pooled_model = _fit_ridge(x_all[train_idx], pooled_y, self.ridge)
            action_models = []
            for action in range(self.action_dim):
                action_mask = train_idx & np.asarray(
                    [int(sample["observed_action_j"]) == action for sample in rows]
                )
                action_models.append(
                    _fit_ridge(
                        x_all[action_mask],
                        [rows[i]["target_return_h"] for i in np.flatnonzero(action_mask)],
                        self.ridge,
                    )
                    if np.count_nonzero(action_mask) >= 2
                    else pooled_model
                )
            indices = np.flatnonzero(test_idx)
            q_hat = np.stack(
                [model.predict(x_all[test_idx]) for model in action_models], axis=1
            )
            for local, row_index in enumerate(indices):
                sample = rows[row_index]
                action = int(sample["observed_action_j"])
                if action < 0 or action >= self.action_dim:
                    raise ValueError(
                        f"observed_action_j must lie in [0, {self.action_dim}), got {action}"
                    )
                pi = np.asarray(sample["policy_probs_j"], dtype=np.float64)
                # The typed action-time record is authoritative whenever it
                # is present.  Compatibility keys remain for diagnostic
                # baselines, but Paper-A calibration must never reconstruct
                # q or b from a differently timed policy array.
                typed_pi = sample.get("target_pi")
                typed_q = sample.get("target_q")
                typed_b = sample.get("target_b")
                if typed_pi is not None:
                    pi_typed = np.asarray(typed_pi, dtype=np.float64)
                    if pi_typed.shape != (self.action_dim,) or not np.allclose(
                        pi, pi_typed, rtol=1e-6, atol=1e-7
                    ):
                        raise ValueError(
                            "policy_probs_j disagrees with typed target_pi"
                        )
                    pi = pi_typed
                if pi.shape != (self.action_dim,) or not np.all(np.isfinite(pi)) or np.any(pi < 0.0):
                    raise ValueError("policy_probs_j must be a finite non-negative action vector")
                valid = np.asarray(
                    sample.get("valid_action_mask", np.ones(self.action_dim)),
                    dtype=bool,
                )
                if valid.shape != (self.action_dim,) or not np.any(valid):
                    raise ValueError("valid_action_mask is malformed")
                if not bool(valid[action]):
                    raise ValueError("observed_action_j is invalid under its recorded mask")
                if np.any(pi[~valid] > 1e-8):
                    raise ValueError(
                        "policy_probs_j must assign zero mass to invalid actions"
                    )
                pi = np.where(valid, pi, 0.0)
                pi = pi / np.clip(pi.sum(), 1e-12, None)
                q_uniform = valid.astype(np.float64) / float(valid.sum())
                if typed_q is not None:
                    q_typed = np.asarray(typed_q, dtype=np.float64)
                    if (
                        q_typed.shape != (self.action_dim,)
                        or not np.all(np.isfinite(q_typed))
                        or np.any(q_typed < 0.0)
                        or np.any(q_typed[~valid] > 1e-8)
                        or not np.allclose(q_typed, q_uniform, rtol=1e-6, atol=1e-7)
                    ):
                        raise ValueError(
                            "target_q must be the recorded fixed rule: uniform over valid actions"
                        )
                    q_uniform = q_typed
                behaviour_raw = float(sample["behaviour_prob_j"])
                if not np.isfinite(behaviour_raw) or behaviour_raw <= 0.0:
                    raise ValueError("behaviour_prob_j must be finite and strictly positive")
                if typed_b is not None:
                    b_typed = np.asarray(typed_b, dtype=np.float64)
                    if (
                        b_typed.shape != (self.action_dim,)
                        or not np.all(np.isfinite(b_typed))
                        or np.any(b_typed < 0.0)
                        or np.any(b_typed[~valid] > 1e-8)
                        or not np.isclose(float(b_typed.sum()), 1.0, rtol=1e-6, atol=1e-7)
                        or not np.isclose(
                            behaviour_raw, float(b_typed[action]), rtol=1e-6, atol=1e-7
                        )
                    ):
                        raise ValueError(
                            "behaviour_prob_j disagrees with typed target_b"
                        )
                    epsilon = sample.get("target_epsilon")
                    if epsilon is not None:
                        epsilon = float(epsilon)
                        if not 0.0 <= epsilon <= 1.0:
                            raise ValueError("target_epsilon must lie in [0, 1]")
                        expected_b = (1.0 - epsilon) * pi + epsilon * q_uniform
                        if not np.allclose(b_typed, expected_b, rtol=1e-5, atol=1e-6):
                            raise ValueError(
                                "typed target_b violates b=(1-epsilon)pi+epsilon q"
                            )
                behaviour = behaviour_raw
                weight = (pi[action] - q_uniform[action]) / behaviour
                if self.iw_clip is not None:
                    weight = float(np.clip(weight, -self.iw_clip, self.iw_clip))
                plugin = float(np.dot(pi - q_uniform, q_hat[local]))
                residual = float(sample["target_return_h"] - q_hat[local, action])
                phi[row_index] = plugin + weight * residual

        valid = np.isfinite(phi)
        if not np.all(valid):
            raise RuntimeError("Cross-fitting did not produce every held-out score")
        self.effect_model = _fit_ridge(x_all, phi, self.ridge)
        self.diagnostics = {
            "n_rows": int(len(rows)),
            "n_episodes": int(len(episodes)),
            "n_folds": int(n_folds),
            "trajectory_leakage_absent": bool(all(leakage_checks)),
            "pseudo_outcome_mean": float(np.mean(phi)),
            "pseudo_outcome_std": float(np.std(phi)),
        }
        return self

    def predict(self, samples):
        if self.effect_model is None:
            raise RuntimeError("fit() must be called before predict()")
        x = np.stack(
            [proxy_context_features(sample, self.action_dim) for sample in samples]
        )
        return self.effect_model.predict(x)
