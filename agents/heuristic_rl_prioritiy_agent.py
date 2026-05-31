import numpy as np

from agents.heuristic_agent import FrameProcessor, HeuristicAgent
from agents.structs import FrameData, GameAction


# RGB values for the ARC-AGI-3 colour palette (indices 0–15, plus 16 for masked status bars)
_COLOR_RGB = [
    (255, 255, 255),  # 0  white
    (204, 204, 204),  # 1  light grey
    (153, 153, 153),  # 2  light grey
    (102, 102, 102),  # 3  grey
    (51,  51,  51),   # 4  dark grey
    (0,   0,   0),    # 5  black
    (255, 0,   0),    # 6
    (0,   255, 0),    # 7
    (250, 61,  50),   # 8  red
    (31,  147, 255),  # 9  blue
    (137, 216, 241),  # 10 light blue
    (255, 221, 0),    # 11 yellow
    (255, 133, 26),   # 12 orange
    (229, 58,  163),  # 13 pink
    (79,  205, 48),   # 14 green
    (163, 86,  214),  # 15 purple
    (0,   0,   0),    # 16 masked status bar — treated as black
]

# Precomputed HSV saturation for each colour index: (max-min)/max, 0 for pure black
_SATURATION: np.ndarray = np.array([
    (max(r, g, b) - min(r, g, b)) / max(r, g, b) if max(r, g, b) > 0 else 0.0
    for r, g, b in _COLOR_RGB
], dtype=float)


class FrameProcessorRL(FrameProcessor):
    """FrameProcessor with RL/MAML/NN-based learnable segment priority assignment.

    features per segment:
        [saturation, closeness_to_medium, is_status_bar, log1p_twins]  (4 floats)
    reward signal: +1 if clicking that segment caused a frame transition, -1 otherwise
    output:        5 priority groups (group 0 = highest priority)
    """

    SAC_RAW_FEATURE_DIM = 10

    def __init__(self, priority_mode: str = "heuristic") -> None:
        super().__init__()
        self.priority_mode = priority_mode
        self.last_features_list: list[list[float]] = []
        self.last_features_list: list[list[float]] = []

        # --- Vanilla RL: linear policy over 4 features ---
        # Warm-start approximates heuristic ordering:
        # [saturation, closeness_to_medium, is_status_bar, log1p_twins]
        self.rl_weights = np.array([0.4, 0.6, -1.0, -0.2])
        self.rl_lr: float = 0.05

        # --- MAML (first-order FOMAML): linear policy ---
        self.maml_meta_weights = np.array([0.4, 0.6, -1.0, -0.2])
        self.maml_task_weights = np.array([0.4, 0.6, -1.0, -0.2])
        self.maml_meta_lr: float = 0.01
        self.maml_inner_lr: float = 0.05
        self.maml_task_experience: list[tuple[np.ndarray, float]] = []

        # --- Neural net: 4 → 8 (ReLU) → 1 (sigmoid) ---
        rng = np.random.default_rng(42)
        self.nn_W1 = rng.normal(0.0, 0.1, (8, 4))
        self.nn_b1 = np.zeros(8)
        self.nn_W2 = rng.normal(0.0, 0.1, (1, 8))
        self.nn_b2 = np.zeros(1)
        self.nn_lr: float = 0.05

        # --- SAC: contextual discrete Soft Actor-Critic over click segments ---
        # A frame is treated as a contextual bandit state whose actions are the
        # candidate segments. The policy ranks segments, while twin linear Q
        # critics learn the transition reward for each segment feature vector.
        self.sac_feature_dim = self.SAC_RAW_FEATURE_DIM + 1  # rich features + bias
        self.sac_actor_weights = rng.normal(0.0, 0.05, self.sac_feature_dim)
        self.sac_q1_weights = np.zeros(self.sac_feature_dim, dtype=float)
        self.sac_q2_weights = np.zeros(self.sac_feature_dim, dtype=float)
        self.sac_q1_weights[:3] = [0.25, 0.45, -0.9]
        self.sac_q2_weights[:3] = [0.15, 0.35, -0.7]
        self.sac_actor_lr: float = 0.03
        self.sac_critic_lr: float = 0.08
        self.sac_alpha: float = 0.2
        self.sac_l2: float = 1e-4
        self.sac_update_count: int = 0

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        if logits.size == 0:
            return logits
        shifted = logits - np.max(logits)
        exp_logits = np.exp(np.clip(shifted, -20.0, 20.0))
        return exp_logits / np.sum(exp_logits)

    @classmethod
    def _augment_features(
        cls, features: list[list[float]] | list[float] | np.ndarray
    ) -> np.ndarray:
        F = np.array(features, dtype=float)
        if F.size == 0:
            return np.empty((0, cls.SAC_RAW_FEATURE_DIM + 1), dtype=float)
        if F.ndim == 1:
            F = F.reshape(1, -1)
        if F.shape[1] < cls.SAC_RAW_FEATURE_DIM:
            padding = np.zeros((F.shape[0], cls.SAC_RAW_FEATURE_DIM - F.shape[1]))
            F = np.concatenate([F, padding], axis=1)
        elif F.shape[1] > cls.SAC_RAW_FEATURE_DIM:
            F = F[:, : cls.SAC_RAW_FEATURE_DIM]
        bias = np.ones((F.shape[0], 1), dtype=float)
        return np.concatenate([F, bias], axis=1)

    def _segment_to_features(self, seg: dict) -> list[float]:
        x1, y1, x2, y2 = seg["bounding_box"]
        x_w = x2 - x1 + 1
        y_w = y2 - y1 + 1
        is_salient = float(seg["color"] in self.salient_color)
        is_medium = float(
            self.minimal_width <= x_w <= self.maximal_width
            and self.minimal_width <= y_w <= self.maximal_width
        )
        is_status = float(seg["color"] == self.status_bar_color)
        area_norm = float(seg.get("area", x_w * y_w)) / float(
            self.frame_shape[0] * self.frame_shape[1]
        )
        twins_norm = min(float(seg.get("number_of_twins", 0)), 8.0) / 8.0
        return [
            is_salient,
            is_medium,
            is_status,
            x_w / float(self.frame_shape[1]),
            y_w / float(self.frame_shape[0]),
            area_norm,
            float(seg.get("is_rectangle", False)),
            twins_norm,
            ((x1 + x2) / 2.0) / float(self.frame_shape[1] - 1),
            ((y1 + y2) / 2.0) / float(self.frame_shape[0] - 1),
        ]

    def _scores_to_groups(self, scores: np.ndarray, n_groups: int = 5) -> list[set[int]]:
        """Rank segments by score descending (high score = group 0) and split evenly."""
        n = len(scores)
        if n == 0:
            return [set() for _ in range(n_groups)]
        sorted_ids = np.argsort(scores)[::-1]
        groups: list[set[int]] = [set() for _ in range(n_groups)]
        for rank, seg_id in enumerate(sorted_ids):
            g = min(rank * n_groups // n, n_groups - 1)
            groups[g].add(int(seg_id))
        return groups

    # ------------------------------------------------------------------
    # Override: collect features then dispatch
    # ------------------------------------------------------------------

    def frame_segments_to_action_groups(
        self, frame_segments: list[dict], n_groups: int
    ) -> list[set[int]]:
        assert n_groups == 5, "Only 5 groups are supported"
        features_list: list[list[float]] = []
        for seg in frame_segments:
            rich_features = self._segment_to_features(seg)
            if self.priority_mode == "sac":
                features_list.append(rich_features)
            else:
                features_list.append(rich_features[:3])
        self.last_features_list = features_list
        return self.create_priority_groups(features_list)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def create_priority_groups(self, features: list[list[float]]) -> list[set[int]]:
        if self.priority_mode == "heuristic":
            return self.create_priority_groups_heuristic(features)
        elif self.priority_mode == "vanilla_rl":
            return self.create_priority_groups_vanilla_rl(features)
        elif self.priority_mode == "maml":
            return self.create_priority_groups_maml(features)
        elif self.priority_mode == "nn":
            return self.create_priority_groups_nn(features)
        elif self.priority_mode == "sac":
            return self.create_priority_groups_sac(features)
        raise ValueError(f"Unknown priority_mode: {self.priority_mode!r}")

    # ------------------------------------------------------------------
    # Heuristic baseline (same logic as heuristic_agent.py)
    # ------------------------------------------------------------------

    def create_priority_groups_heuristic(
        self, features: list[list[float]]
    ) -> list[set[int]]:
        groups: list[set[int]] = [set() for _ in range(5)]
        for seg_id, (is_salient, is_medium, is_status) in enumerate(features):
            if is_salient and is_medium:
                groups[0].add(seg_id)
            elif is_medium:
                groups[1].add(seg_id)
            elif is_salient:
                groups[2].add(seg_id)
            elif not is_status:
                groups[3].add(seg_id)
            else:
                groups[4].add(seg_id)
        return groups

    # ------------------------------------------------------------------
    # Vanilla policy-gradient RL
    # ------------------------------------------------------------------

    def create_priority_groups_vanilla_rl(
        self, features: list[list[float]]
    ) -> list[set[int]]:
        """
        Use RL with dense +-1 signal of frame changes via Vanilla RL

        Policy: score_i = sigmoid(w · f_i).  Segments sorted by score descending
        and split into 5 priority groups.  Weights are updated online via
        REINFORCE-style gradient: Δw = α · r · f · σ(1 − σ).
        """
        if not features:
            return [set() for _ in range(5)]
        F = np.array(features, dtype=float)
        scores = self._sigmoid(F @ self.rl_weights)
        return self._scores_to_groups(scores)

    def _update_vanilla_rl(self, seg_features: list[float], reward: float) -> None:
        f = np.array(seg_features, dtype=float)
        s = self._sigmoid(f @ self.rl_weights)
        self.rl_weights += self.rl_lr * reward * f * s * (1.0 - s)

    # ------------------------------------------------------------------
    # MAML (first-order / FOMAML)
    # ------------------------------------------------------------------

    def create_priority_groups_maml(
        self, features: list[list[float]]
    ) -> list[set[int]]:
        """
        Use RL with dense +-1 signal of frame changes via MAML

        Uses task-adapted fast weights φ (initialised from meta-weights θ at the
        start of each level) for the per-segment priority score.  After each level,
        an outer meta-gradient step updates θ using the task experience, so the
        agent learns an initialisation that adapts quickly to new levels.
        """
        if not features:
            return [set() for _ in range(5)]
        F = np.array(features, dtype=float)
        scores = self._sigmoid(F @ self.maml_task_weights)
        return self._scores_to_groups(scores)

    def _update_maml_inner(self, seg_features: list[float], reward: float) -> None:
        """Inner-loop gradient step on task (fast) weights."""
        f = np.array(seg_features, dtype=float)
        self.maml_task_experience.append((f.copy(), reward))
        s = self._sigmoid(f @ self.maml_task_weights)
        self.maml_task_weights += self.maml_inner_lr * reward * f * s * (1.0 - s)

    def on_new_level(self) -> None:
        """Outer meta-update (FOMAML) when a new level begins; resets task weights."""
        if self.priority_mode != "maml":
            return
        if self.maml_task_experience:
            meta_grad = np.zeros_like(self.maml_meta_weights)
            for f, r in self.maml_task_experience:
                s = self._sigmoid(f @ self.maml_task_weights)
                meta_grad += r * f * s * (1.0 - s)
            meta_grad /= len(self.maml_task_experience)
            self.maml_meta_weights += self.maml_meta_lr * meta_grad
        self.maml_task_weights = self.maml_meta_weights.copy()
        self.maml_task_experience = []

    # ------------------------------------------------------------------
    # Neural network (2-layer MLP with online backprop)
    # ------------------------------------------------------------------

    def create_priority_groups_nn(
        self, features: list[list[float]]
    ) -> list[set[int]]:
        """
        Use neural net to predict P(transition|φ(segment))

        Architecture: 4 → Linear → ReLU → 8 → Linear → Sigmoid → priority score.
        Trained online with binary cross-entropy against the ±1 transition signal
        (converted to 0/1 targets).  Segments ranked by predicted transition
        probability and split into 5 priority groups.
        """
        if not features:
            return [set() for _ in range(5)]
        F = np.array(features, dtype=float)
        H = np.maximum(0.0, F @ self.nn_W1.T + self.nn_b1)
        scores = self._sigmoid((H @ self.nn_W2.T + self.nn_b2).ravel())
        return self._scores_to_groups(scores)

    def _update_nn(self, seg_features: list[float], reward: float) -> None:
        f = np.array(seg_features, dtype=float)             # (3,)
        # Forward
        h_pre = f @ self.nn_W1.T + self.nn_b1              # (8,)
        h = np.maximum(0.0, h_pre)                          # (8,)
        score = float(self._sigmoid((h @ self.nn_W2.T + self.nn_b2).ravel())[0])
        target = (reward + 1.0) / 2.0
        d_out = score - target
        d_W2 = d_out * h[np.newaxis, :]
        d_b2 = np.array([d_out])
        d_h = d_out * self.nn_W2.squeeze(0)
        d_h_pre = d_h * (h_pre > 0).astype(float)
        d_W1 = d_h_pre[:, np.newaxis] * f[np.newaxis, :]
        d_b1 = d_h_pre
        self.nn_W1 -= self.nn_lr * d_W1
        self.nn_b1 -= self.nn_lr * d_b1
        self.nn_W2 -= self.nn_lr * d_W2
        self.nn_b2 -= self.nn_lr * d_b2

    # ------------------------------------------------------------------
    # Soft Actor-Critic priority model
    # ------------------------------------------------------------------

    def create_priority_groups_sac(
        self, features: list[list[float]]
    ) -> list[set[int]]:
        """
        Rank segments with a lightweight discrete SAC policy.

        Each frame is a contextual bandit: actions are connected components,
        rewards are the observed transition outcomes, and there is no bootstrapped
        next-state target because the graph explorer owns long-horizon planning.
        The actor keeps entropy in the score so uncertain alternatives stay alive,
        while the clipped double critics stabilize value estimates.
        """
        if not features:
            return [set() for _ in range(5)]
        F = self._augment_features(features)
        logits = F @ self.sac_actor_weights
        policy = self._softmax(logits)
        q_values = np.minimum(F @ self.sac_q1_weights, F @ self.sac_q2_weights)
        entropy_bonus = -self.sac_alpha * np.log(np.clip(policy, 1e-8, 1.0))
        scores = q_values + entropy_bonus
        return self._scores_to_groups(scores)

    def _update_sac(
        self,
        seg_features: list[float],
        reward: float,
        frame_features: list[list[float]] | None = None,
        action_index: int | None = None,
    ) -> None:
        clipped_reward = float(np.clip(reward, -1.0, 1.0))
        f = self._augment_features(seg_features)[0]

        # Critic update: contextual-bandit Bellman target is the observed reward.
        q1 = float(f @ self.sac_q1_weights)
        q2 = float(f @ self.sac_q2_weights)
        td1 = clipped_reward - q1
        td2 = clipped_reward - q2
        self.sac_q1_weights += self.sac_critic_lr * (
            td1 * f - self.sac_l2 * self.sac_q1_weights
        )
        self.sac_q2_weights += self.sac_critic_lr * (
            td2 * f - self.sac_l2 * self.sac_q2_weights
        )

        # Actor update: minimize E_pi[alpha log pi(a|s) - min(Q1,Q2)].
        if frame_features and action_index is not None:
            F = self._augment_features(frame_features)
        else:
            F = f.reshape(1, -1)

        if len(F) > 1:
            logits = F @ self.sac_actor_weights
            policy = self._softmax(logits)
            q_values = np.minimum(F @ self.sac_q1_weights, F @ self.sac_q2_weights)
            policy_objective = self.sac_alpha * (
                np.log(np.clip(policy, 1e-8, 1.0)) + 1.0
            ) - q_values
            baseline = float(np.sum(policy * policy_objective))
            grad_logits = policy * (policy_objective - baseline)
            grad_actor = F.T @ grad_logits
            self.sac_actor_weights -= self.sac_actor_lr * (
                grad_actor + self.sac_l2 * self.sac_actor_weights
            )
        else:
            # Degenerate one-action frame: train the actor toward good/bad outcomes
            # so future same-shaped frames still benefit from the signal.
            prob = float(self._sigmoid(f @ self.sac_actor_weights))
            target = 1.0 if clipped_reward > 0 else 0.0
            self.sac_actor_weights -= self.sac_actor_lr * (
                (prob - target) * f + self.sac_l2 * self.sac_actor_weights
            )

        self.sac_update_count += 1

    # ------------------------------------------------------------------
    # Unified update entry point (called from HeuristicRLAgent)
    # ------------------------------------------------------------------

    def record_outcome(
        self,
        seg_features: list[float],
        reward: float,
        frame_features: list[list[float]] | None = None,
        action_index: int | None = None,
    ) -> None:
        """Update weights given the observed transition outcome for one segment."""
        if self.priority_mode == "vanilla_rl":
            self._update_vanilla_rl(seg_features, reward)
        elif self.priority_mode == "maml":
            self._update_maml_inner(seg_features, reward)
        elif self.priority_mode == "nn":
            self._update_nn(seg_features, reward)
        elif self.priority_mode == "sac":
            self._update_sac(
                seg_features,
                reward,
                frame_features=frame_features,
                action_index=action_index,
            )
        # heuristic: no parameters to update


# ---------------------------------------------------------------------------
# Agent classes
# ---------------------------------------------------------------------------

class HeuristicRLAgent(HeuristicAgent):
    """HeuristicAgent that replaces FrameProcessor with FrameProcessorRL.

    Override PRIORITY_MODE in subclasses to select the learning algorithm.
    The choose_action override feeds the ±1 transition reward back to the
    FrameProcessorRL after each step so the model can learn online.
    """

    PRIORITY_MODE: str = "heuristic"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.frame_processor = FrameProcessorRL(priority_mode=self.PRIORITY_MODE)

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        prev_hash = self.last_hashed_frame
        prev_action = self.last_action
        prev_features = list(self.frame_processor.last_features_list)
        was_level_up = self.level_up

        action = super().choose_action(frames, latest_frame)

        if (
            prev_hash is not None
            and prev_action is not None
            and prev_action < len(prev_features)
        ):
            result_arr = self.hashed_frame2action_results.get(prev_hash)
            if result_arr is not None:
                reward = float(result_arr[prev_action])
                if reward != 0.0:
                    self.frame_processor.record_outcome(
                        prev_features[prev_action],
                        reward,
                        frame_features=prev_features,
                        action_index=prev_action,
                    )

        if was_level_up:
            self.frame_processor.on_new_level()

        return action


class HeuristicRLVanillaAgent(HeuristicRLAgent):
    """HeuristicRLAgent using vanilla policy-gradient RL for segment priority."""
    PRIORITY_MODE = "vanilla_rl"


class HeuristicRLMAMLAgent(HeuristicRLAgent):
    """HeuristicRLAgent using first-order MAML for segment priority."""
    PRIORITY_MODE = "maml"


class HeuristicRLNNAgent(HeuristicRLAgent):
    """HeuristicRLAgent using a 2-layer MLP for segment priority."""
    PRIORITY_MODE = "nn"


class HeuristicRLSACAgent(HeuristicRLAgent):
    """HeuristicRLAgent using contextual discrete SAC for segment priority."""
    PRIORITY_MODE = "sac"
