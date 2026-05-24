import numpy as np

from agents.heuristic_agent import FrameProcessor, HeuristicAgent
from agents.structs import FrameData, GameAction


class FrameProcessorRL(FrameProcessor):
    """FrameProcessor with RL/MAML/NN-based learnable segment priority assignment.

    features per segment: [is_salient, is_medium_width, is_status_bar]  (3 bools)
    reward signal:        +1 if clicking that segment caused a frame transition, -1 otherwise
    output:               5 priority groups (group 0 = highest priority)
    """

    def __init__(self, priority_mode: str = "heuristic") -> None:
        super().__init__()
        self.priority_mode = priority_mode
        self.last_features_list: list[list[bool]] = []

        # --- Vanilla RL: linear policy over 3 features ---
        self.rl_weights = np.zeros(3)
        self.rl_lr: float = 0.05

        # --- MAML (first-order FOMAML): linear policy ---
        self.maml_meta_weights = np.zeros(3)
        self.maml_task_weights = np.zeros(3)   # fast weights, reset each level
        self.maml_meta_lr: float = 0.01
        self.maml_inner_lr: float = 0.05
        self.maml_task_experience: list[tuple[np.ndarray, float]] = []

        # --- Neural net: 3 → 8 (ReLU) → 1 (sigmoid) ---
        rng = np.random.default_rng(42)
        self.nn_W1 = rng.normal(0.0, 0.1, (8, 3))
        self.nn_b1 = np.zeros(8)
        self.nn_W2 = rng.normal(0.0, 0.1, (1, 8))
        self.nn_b2 = np.zeros(1)
        self.nn_lr: float = 0.05

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))

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
    # Override: collect all features first, then call create_priority_groups once
    # ------------------------------------------------------------------

    def frame_segments_to_action_groups(
        self, frame_segments: list[dict], n_groups: int
    ) -> list[set[int]]:
        assert n_groups == 5, "Only 5 groups are supported"
        features_list: list[list[bool]] = []
        for seg in frame_segments:
            x_w = seg["bounding_box"][2] - seg["bounding_box"][0] + 1
            y_w = seg["bounding_box"][3] - seg["bounding_box"][1] + 1
            is_salient = seg["color"] in self.salient_color
            is_medium = (
                self.minimal_width <= x_w <= self.maximal_width
                and self.minimal_width <= y_w <= self.maximal_width
            )
            is_status = seg["color"] == self.status_bar_color
            features_list.append([is_salient, is_medium, is_status])
        self.last_features_list = features_list
        return self.create_priority_groups(features_list)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def create_priority_groups(self, features: list[list[bool]]) -> list[set[int]]:
        if self.priority_mode == "heuristic":
            return self.create_priority_groups_heuristic(features)
        elif self.priority_mode == "vanilla_rl":
            return self.create_priority_groups_vanilla_rl(features)
        elif self.priority_mode == "maml":
            return self.create_priority_groups_maml(features)
        elif self.priority_mode == "nn":
            return self.create_priority_groups_nn(features)
        raise ValueError(f"Unknown priority_mode: {self.priority_mode!r}")

    # ------------------------------------------------------------------
    # Heuristic baseline (same logic as heuristic_agent.py)
    # ------------------------------------------------------------------

    def create_priority_groups_heuristic(
        self, features: list[list[bool]]
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
        self, features: list[list[bool]]
    ) -> list[set[int]]:
        """
        Use RL with dense +-1 signal of frame changes via Vanilla RL

        Policy: score_i = sigmoid(w · f_i).  Segments sorted by score descending
        and split into 5 priority groups.  Weights are updated online via
        REINFORCE-style gradient: Δw = α · r · f · σ(1 − σ).
        """
        if not features:
            return [set() for _ in range(5)]
        F = np.array(features, dtype=float)                # (N, 3)
        scores = self._sigmoid(F @ self.rl_weights)        # (N,)
        return self._scores_to_groups(scores)

    def _update_vanilla_rl(self, seg_features: list[bool], reward: float) -> None:
        f = np.array(seg_features, dtype=float)
        s = self._sigmoid(f @ self.rl_weights)
        self.rl_weights += self.rl_lr * reward * f * s * (1.0 - s)

    # ------------------------------------------------------------------
    # MAML (first-order / FOMAML)
    # ------------------------------------------------------------------

    def create_priority_groups_maml(
        self, features: list[list[bool]]
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

    def _update_maml_inner(self, seg_features: list[bool], reward: float) -> None:
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
    # Unsupervised neural network (2-layer MLP with online backprop)
    # ------------------------------------------------------------------

    def create_priority_groups_nn(
        self, features: list[list[bool]]
    ) -> list[set[int]]:
        """
        Use unsupervised neural net to do above

        Architecture: 3 → Linear → ReLU → 8 → Linear → Sigmoid → priority score.
        Trained online with binary cross-entropy against the ±1 transition signal
        (converted to 0/1 targets).  Segments ranked by predicted transition
        probability and split into 5 priority groups.
        """
        if not features:
            return [set() for _ in range(5)]
        F = np.array(features, dtype=float)                         # (N, 3)
        H = np.maximum(0.0, F @ self.nn_W1.T + self.nn_b1)         # (N, 8)
        scores = self._sigmoid((H @ self.nn_W2.T + self.nn_b2).ravel())  # (N,)
        return self._scores_to_groups(scores)

    def _update_nn(self, seg_features: list[bool], reward: float) -> None:
        f = np.array(seg_features, dtype=float)             # (3,)
        # Forward
        h_pre = f @ self.nn_W1.T + self.nn_b1              # (8,)
        h = np.maximum(0.0, h_pre)                          # (8,)
        score = float(self._sigmoid((h @ self.nn_W2.T + self.nn_b2).ravel())[0])
        target = (reward + 1.0) / 2.0                       # ±1 → 0/1
        d_out = score - target                              # scalar BCE gradient
        # Backprop through output layer
        d_W2 = d_out * h[np.newaxis, :]                    # (1, 8)
        d_b2 = np.array([d_out])
        # Backprop through hidden layer
        d_h = d_out * self.nn_W2.squeeze(0)                # (8,)
        d_h_pre = d_h * (h_pre > 0).astype(float)          # (8,) ReLU mask
        d_W1 = d_h_pre[:, np.newaxis] * f[np.newaxis, :]   # (8, 3)
        d_b1 = d_h_pre
        # Gradient-descent update
        self.nn_W1 -= self.nn_lr * d_W1
        self.nn_b1 -= self.nn_lr * d_b1
        self.nn_W2 -= self.nn_lr * d_W2
        self.nn_b2 -= self.nn_lr * d_b2

    # ------------------------------------------------------------------
    # Unified update entry point (called from HeuristicRLAgent)
    # ------------------------------------------------------------------

    def record_outcome(self, seg_features: list[bool], reward: float) -> None:
        """Update weights given the observed transition outcome for one segment."""
        if self.priority_mode == "vanilla_rl":
            self._update_vanilla_rl(seg_features, reward)
        elif self.priority_mode == "maml":
            self._update_maml_inner(seg_features, reward)
        elif self.priority_mode == "nn":
            self._update_nn(seg_features, reward)
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
        # Swap in the learnable processor (super().__init__ already created one,
        # so we simply replace it)
        self.frame_processor = FrameProcessorRL(priority_mode=self.PRIORITY_MODE)

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        # Snapshot state BEFORE super() processes the current transition
        prev_hash = self.last_hashed_frame
        prev_action = self.last_action
        # Shallow copy so mutation inside super() doesn't affect us
        prev_features = list(self.frame_processor.last_features_list)
        was_level_up = self.level_up

        action = super().choose_action(frames, latest_frame)

        # super() has now written ±1 into hashed_frame2action_results for the
        # previous (hash, action) pair.  Feed that reward to the RL model.
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
                        prev_features[prev_action], reward
                    )

        # When a new level starts, run the MAML outer update and reset task weights
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
