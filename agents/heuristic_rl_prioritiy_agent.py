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
    """FrameProcessor with RL/MAML/NN/SAC/PPO-based learnable segment priority assignment.

    features per segment:
        [saturation, closeness_to_medium, is_status_bar, log1p_twins]  (4 floats)
    reward signal: +1 if clicking that segment caused a frame transition, -1 otherwise
    output:        5 priority groups (group 0 = highest priority)
    """

    # Closeness-to-medium: Gaussian in log-area space centred at area=64
    # (geometric mean of the 2–32 px medium range in each dimension: sqrt(2*32)^2 = 64)
    _LOG_IDEAL_AREA: float = float(np.log1p(64.0))
    _SIGMA: float = 2.5

    def __init__(self, priority_mode: str = "heuristic") -> None:
        super().__init__()
        self.priority_mode = priority_mode
        self.last_features_list: list[list[float]] = []

        # --- Vanilla RL: linear policy over 4 features ---
        # Warm-start approximates heuristic ordering:
        # [saturation, closeness_to_medium, is_status_bar, log1p_twins]
        self.rl_weights = np.array([0.2, 0.8, -1.0, -0.2])
        self.rl_lr: float = 0.05

        # --- MAML (first-order FOMAML): linear policy ---
        self.maml_meta_weights = np.array([0.2, 0.8, -1.0, -0.2])
        self.maml_task_weights = np.array([0.2, 0.8, -1.0, -0.2])
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

        # --- SAC: Soft Actor-Critic (off-policy, twin critics, entropy regularisation) ---
        # actor 4→8 ReLU→1 sigmoid;  Q1/Q2/Q1t/Q2t: 4→8 ReLU→1 linear
        rng_sac = np.random.default_rng(43)
        self.sac_actor_W1 = rng_sac.normal(0.0, 0.1, (8, 4))
        self.sac_actor_b1 = np.zeros(8)
        self.sac_actor_W2 = rng_sac.normal(0.0, 0.1, (1, 8))
        self.sac_actor_b2 = np.zeros(1)
        self.sac_q1_W1 = rng_sac.normal(0.0, 0.1, (8, 4))
        self.sac_q1_b1 = np.zeros(8)
        self.sac_q1_W2 = rng_sac.normal(0.0, 0.1, (1, 8))
        self.sac_q1_b2 = np.zeros(1)
        self.sac_q2_W1 = rng_sac.normal(0.0, 0.1, (8, 4))
        self.sac_q2_b1 = np.zeros(8)
        self.sac_q2_W2 = rng_sac.normal(0.0, 0.1, (1, 8))
        self.sac_q2_b2 = np.zeros(1)
        self.sac_q1t_W1 = self.sac_q1_W1.copy()
        self.sac_q1t_b1 = self.sac_q1_b1.copy()
        self.sac_q1t_W2 = self.sac_q1_W2.copy()
        self.sac_q1t_b2 = self.sac_q1_b2.copy()
        self.sac_q2t_W1 = self.sac_q2_W1.copy()
        self.sac_q2t_b1 = self.sac_q2_b1.copy()
        self.sac_q2t_W2 = self.sac_q2_W2.copy()
        self.sac_q2t_b2 = self.sac_q2_b2.copy()
        self.sac_log_alpha: float = float(np.log(0.2))
        self.sac_alpha_lr: float = 0.003
        self.sac_target_entropy: float = float(np.log(2.0))  # H[Bernoulli(0.5)]
        self.sac_replay: list[tuple[np.ndarray, float]] = []
        self.sac_replay_capacity: int = 1000
        self.sac_batch_size: int = 16
        self.sac_lr: float = 0.01
        self.sac_tau: float = 0.005
        self.sac_update_freq: int = 1
        self.sac_steps: int = 0

        # --- PPO: Proximal Policy Optimisation (on-policy, actor-critic, clipped surrogate) ---
        # Actor  4→8 ReLU→1 sigmoid  — outputs p(click | φ)
        # Critic 4→8 ReLU→1 linear   — outputs V(φ), used as GAE baseline
        rng_ppo = np.random.default_rng(44)
        self.ppo_actor_W1  = rng_ppo.normal(0.0, 0.1, (8, 4))
        self.ppo_actor_b1  = np.zeros(8)
        self.ppo_actor_W2  = rng_ppo.normal(0.0, 0.1, (1, 8))
        self.ppo_actor_b2  = np.zeros(1)
        self.ppo_critic_W1 = rng_ppo.normal(0.0, 0.1, (8, 4))
        self.ppo_critic_b1 = np.zeros(8)
        self.ppo_critic_W2 = rng_ppo.normal(0.0, 0.1, (1, 8))
        self.ppo_critic_b2 = np.zeros(1)
        self.ppo_lr:          float = 0.01
        self.ppo_clip_eps:    float = 0.2   # surrogate clip ratio ε
        self.ppo_gamma:       float = 0.99  # discount factor
        self.ppo_lam:         float = 0.95  # GAE-λ
        self.ppo_c_value:     float = 0.5   # value-loss coefficient
        self.ppo_c_entropy:   float = 0.01  # entropy bonus coefficient
        self.ppo_epochs:      int   = 4     # gradient epochs per rollout flush
        self.ppo_batch_size:  int   = 8     # mini-batch size within an epoch
        self.ppo_buffer_size: int   = 32    # flush rollout after this many steps
        # Each entry: (features, log_prob_old, reward, value_old)
        self.ppo_buffer: list[tuple[np.ndarray, float, float, float]] = []

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    @classmethod
    def _segment_features(cls, seg: dict) -> list[float]:
        """Return the 4-dimensional feature vector for one segment."""
        color   = seg["color"]
        area    = seg["area"]
        n_twins = seg["number_of_twins"]

        saturation = float(_SATURATION[min(color, 16)])
        closeness  = float(np.exp(-abs(np.log1p(area) - cls._LOG_IDEAL_AREA) / cls._SIGMA))
        is_status  = float(color == 16)
        log_twins  = float(np.log1p(n_twins))

        return [saturation, closeness, is_status, log_twins]

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
    # Override: collect features then dispatch
    # ------------------------------------------------------------------

    def frame_segments_to_action_groups(
        self, frame_segments: list[dict], n_groups: int
    ) -> list[set[int]]:
        assert n_groups == 5, "Only 5 groups are supported"

        if self.priority_mode == "heuristic":
            # Heuristic mode uses the parent-class rule-based logic unchanged.
            self.last_features_list = []
            return super().frame_segments_to_action_groups(frame_segments, n_groups)

        features_list = [self._segment_features(seg) for seg in frame_segments]
        self.last_features_list = features_list
        return self.create_priority_groups(features_list)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def create_priority_groups(self, features: list[list[float]]) -> list[set[int]]:
        if self.priority_mode == "vanilla_rl":
            return self.create_priority_groups_vanilla_rl(features)
        elif self.priority_mode == "maml":
            return self.create_priority_groups_maml(features)
        elif self.priority_mode == "nn":
            return self.create_priority_groups_nn(features)
        elif self.priority_mode == "sac":
            return self.create_priority_groups_sac(features)
        elif self.priority_mode == "ppo":
            return self.create_priority_groups_ppo(features)
        raise ValueError(f"Unknown priority_mode: {self.priority_mode!r}")

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
        f = np.array(seg_features, dtype=float)
        h_pre = f @ self.nn_W1.T + self.nn_b1
        h = np.maximum(0.0, h_pre)
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
    # Soft Actor-Critic (SAC)
    # ------------------------------------------------------------------

    def _sac_q_step(
        self,
        W1: np.ndarray, b1: np.ndarray,
        W2: np.ndarray, b2: np.ndarray,
        f: np.ndarray, target: float,
    ) -> None:
        """One in-place MSE gradient step on a 4→8 ReLU→1 Q-network."""
        h_pre = f @ W1.T + b1
        h = np.maximum(0.0, h_pre)
        d = float((h @ W2.T + b2).ravel()[0]) - target
        dh = d * W2.squeeze(0) * (h_pre > 0).astype(float)
        W2 -= self.sac_lr * d * h[np.newaxis, :]
        b2 -= self.sac_lr * np.array([d])
        W1 -= self.sac_lr * dh[:, np.newaxis] * f[np.newaxis, :]
        b1 -= self.sac_lr * dh

    def _sac_q_eval(
        self,
        W1: np.ndarray, b1: np.ndarray,
        W2: np.ndarray, b2: np.ndarray,
        f: np.ndarray,
    ) -> float:
        h = np.maximum(0.0, f @ W1.T + b1)
        return float((h @ W2.T + b2).ravel()[0])

    def create_priority_groups_sac(
        self, features: list[list[float]]
    ) -> list[set[int]]:
        """
        Use Soft Actor-Critic to score segments.

        Actor (4→8 ReLU→1 sigmoid) is trained to maximise min(Q1, Q2)(f) with
        a Bernoulli entropy bonus (temperature α).  Q1/Q2 are reward predictors
        updated from a replay buffer; soft target copies Q1t/Q2t stabilise training.
        """
        if not features:
            return [set() for _ in range(5)]
        F = np.array(features, dtype=float)
        H = np.maximum(0.0, F @ self.sac_actor_W1.T + self.sac_actor_b1)
        scores = self._sigmoid((H @ self.sac_actor_W2.T + self.sac_actor_b2).ravel())
        return self._scores_to_groups(scores)

    def _update_sac(self, seg_features: list[float], reward: float) -> None:
        f = np.array(seg_features, dtype=float)
        self.sac_replay.append((f.copy(), reward))
        if len(self.sac_replay) > self.sac_replay_capacity:
            self.sac_replay.pop(0)
        self.sac_steps += 1
        if (len(self.sac_replay) < self.sac_batch_size
                or self.sac_steps % self.sac_update_freq != 0):
            return

        indices = np.random.choice(len(self.sac_replay), self.sac_batch_size, replace=False)
        alpha = float(np.exp(self.sac_log_alpha))
        total_entropy = 0.0

        for i in indices:
            f_b, r_b = self.sac_replay[int(i)]

            # Actor forward
            h_ap = f_b @ self.sac_actor_W1.T + self.sac_actor_b1
            h_a = np.maximum(0.0, h_ap)
            p_a = float(self._sigmoid((h_a @ self.sac_actor_W2.T + self.sac_actor_b2).ravel())[0])
            pc = np.clip(p_a, 1e-6, 1.0 - 1e-6)
            total_entropy += -pc * np.log(pc) - (1.0 - pc) * np.log(1.0 - pc)

            # Twin critic updates
            self._sac_q_step(self.sac_q1_W1, self.sac_q1_b1, self.sac_q1_W2, self.sac_q1_b2, f_b, r_b)
            self._sac_q_step(self.sac_q2_W1, self.sac_q2_b1, self.sac_q2_W2, self.sac_q2_b2, f_b, r_b)

            # Actor update: minimise (p − σ(min_Q))² − α·H(p)
            q1 = self._sac_q_eval(self.sac_q1_W1, self.sac_q1_b1, self.sac_q1_W2, self.sac_q1_b2, f_b)
            q2 = self._sac_q_eval(self.sac_q2_W1, self.sac_q2_b1, self.sac_q2_W2, self.sac_q2_b2, f_b)
            a_tgt = float(self._sigmoid(np.array([min(q1, q2)]))[0])
            d_dp = 2.0 * (p_a - a_tgt) - alpha * float(np.log((1.0 - pc) / pc))
            d_dpre = d_dp * p_a * (1.0 - p_a)
            d_ha = d_dpre * self.sac_actor_W2.squeeze(0) * (h_ap > 0).astype(float)
            self.sac_actor_W2 -= self.sac_lr * d_dpre * h_a[np.newaxis, :]
            self.sac_actor_b2 -= self.sac_lr * np.array([d_dpre])
            self.sac_actor_W1 -= self.sac_lr * d_ha[:, np.newaxis] * f_b[np.newaxis, :]
            self.sac_actor_b1 -= self.sac_lr * d_ha

        # Temperature update
        self.sac_log_alpha = float(np.clip(
            self.sac_log_alpha + self.sac_alpha_lr * (total_entropy / self.sac_batch_size - self.sac_target_entropy),
            -5.0, 2.0,
        ))

        # Soft-update target Q-networks
        tau = self.sac_tau
        self.sac_q1t_W1 = tau * self.sac_q1_W1 + (1 - tau) * self.sac_q1t_W1
        self.sac_q1t_b1 = tau * self.sac_q1_b1 + (1 - tau) * self.sac_q1t_b1
        self.sac_q1t_W2 = tau * self.sac_q1_W2 + (1 - tau) * self.sac_q1t_W2
        self.sac_q1t_b2 = tau * self.sac_q1_b2 + (1 - tau) * self.sac_q1t_b2
        self.sac_q2t_W1 = tau * self.sac_q2_W1 + (1 - tau) * self.sac_q2t_W1
        self.sac_q2t_b1 = tau * self.sac_q2_b1 + (1 - tau) * self.sac_q2t_b1
        self.sac_q2t_W2 = tau * self.sac_q2_W2 + (1 - tau) * self.sac_q2t_W2
        self.sac_q2t_b2 = tau * self.sac_q2_b2 + (1 - tau) * self.sac_q2t_b2

    # ------------------------------------------------------------------
    # Proximal Policy Optimisation (PPO)
    # ------------------------------------------------------------------

    def _ppo_actor_forward(
        self, f: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Return (pre-activations h_pre, activations h, probability p)."""
        h_pre = f @ self.ppo_actor_W1.T + self.ppo_actor_b1
        h     = np.maximum(0.0, h_pre)
        p     = float(self._sigmoid((h @ self.ppo_actor_W2.T + self.ppo_actor_b2).ravel())[0])
        return h_pre, h, p

    def _ppo_critic_forward(
        self, f: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Return (pre-activations h_pre, activations h, value V)."""
        h_pre = f @ self.ppo_critic_W1.T + self.ppo_critic_b1
        h     = np.maximum(0.0, h_pre)
        v     = float((h @ self.ppo_critic_W2.T + self.ppo_critic_b2).ravel()[0])
        return h_pre, h, v

    def create_priority_groups_ppo(
        self, features: list[list[float]]
    ) -> list[set[int]]:
        """
        Use Proximal Policy Optimisation to score segments.

        Actor (4→8 ReLU→1 sigmoid) outputs p(click | φ(segment)).  Segments
        ranked by p descending and split into 5 priority groups.  The critic
        (4→8 ReLU→1 linear) provides a value baseline for variance reduction via
        GAE-λ.  Weights are updated from a fixed-size rollout buffer using the
        clipped surrogate objective (ε=0.2), value-function MSE (c_v=0.5), and
        a Bernoulli entropy bonus (c_e=0.01) across K=4 gradient epochs.
        """
        if not features:
            return [set() for _ in range(5)]
        scores = np.array(
            [self._ppo_actor_forward(np.array(f, dtype=float))[2] for f in features]
        )
        return self._scores_to_groups(scores)

    def _update_ppo(self, seg_features: list[float], reward: float) -> None:
        """Buffer one (features, reward) transition; flush when full."""
        f              = np.array(seg_features, dtype=float)
        _, _, p        = self._ppo_actor_forward(f)
        _, _, v        = self._ppo_critic_forward(f)
        lp_old         = float(np.log(np.clip(p, 1e-8, 1.0 - 1e-8)))
        self.ppo_buffer.append((f.copy(), lp_old, reward, v))

        if len(self.ppo_buffer) >= self.ppo_buffer_size:
            self._ppo_flush_and_update()

    def _ppo_flush_and_update(self) -> None:
        """Run full PPO update on the collected rollout, then clear the buffer."""
        if not self.ppo_buffer:
            return

        fs, lps_old, rewards, values = zip(*self.ppo_buffer)
        fs      = list(fs)
        lps_old = np.array(lps_old, dtype=float)
        rewards = np.array(rewards,  dtype=float)
        values  = np.array(values,   dtype=float)
        n       = len(rewards)

        # ── GAE-λ advantage computation ───────────────────────────────────────
        # Each segment interaction is treated as terminal, so the bootstrap
        # value after the final step is 0.
        advantages = np.zeros(n, dtype=float)
        gae = 0.0
        for t in reversed(range(n)):
            next_v = values[t + 1] if t + 1 < n else 0.0
            delta  = rewards[t] + self.ppo_gamma * next_v - values[t]
            gae    = delta + self.ppo_gamma * self.ppo_lam * gae
            advantages[t] = gae

        returns    = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ── K epochs of random mini-batch gradient descent ────────────────────
        idx = np.arange(n)
        for _ in range(self.ppo_epochs):
            np.random.shuffle(idx)
            for start in range(0, n, self.ppo_batch_size):
                for i in idx[start : start + self.ppo_batch_size]:
                    self._ppo_step(
                        fs[i], lps_old[i],
                        float(advantages[i]), float(returns[i]),
                    )

        self.ppo_buffer.clear()

    def _ppo_step(
        self,
        f:      np.ndarray,
        lp_old: float,
        adv:    float,
        ret:    float,
    ) -> None:
        """Single-sample PPO gradient step on actor and critic."""

        # ── Actor ─────────────────────────────────────────────────────────────
        h_ap, h_a, p = self._ppo_actor_forward(f)
        p_c   = float(np.clip(p, 1e-8, 1.0 - 1e-8))
        lp    = float(np.log(p_c))
        ratio = float(np.exp(lp - lp_old))
        clip  = float(np.clip(ratio, 1.0 - self.ppo_clip_eps, 1.0 + self.ppo_clip_eps))

        # Gradient of -min(ratio·A, clip·A) w.r.t. log p
        if abs(ratio * adv) <= abs(clip * adv):
            d_lp = -adv          # unclipped branch: d/d(log p) of -(ratio · adv)
        else:
            d_lp = 0.0           # clipped: no gradient flows

        # Entropy H(p) = -p log p - (1-p) log(1-p)  →  dH/dp = log((1-p)/p)
        d_entropy = float(np.log((1.0 - p_c) / p_c))

        # Combined actor loss gradient: d(-L_clip - c_e·H)/dp
        d_p   = d_lp / p_c - self.ppo_c_entropy * d_entropy
        d_pre = d_p * p * (1.0 - p)   # chain through sigmoid

        g_W2 = d_pre * h_a[np.newaxis, :]
        g_b2 = np.array([d_pre])
        g_h  = d_pre * self.ppo_actor_W2.squeeze(0) * (h_ap > 0).astype(float)
        g_W1 = g_h[:, np.newaxis] * f[np.newaxis, :]
        g_b1 = g_h

        self.ppo_actor_W1 -= self.ppo_lr * g_W1
        self.ppo_actor_b1 -= self.ppo_lr * g_b1
        self.ppo_actor_W2 -= self.ppo_lr * g_W2
        self.ppo_actor_b2 -= self.ppo_lr * g_b2

        # ── Critic MSE: L_v = c_v · (V(f) - ret)² ───────────────────────────
        h_cp, h_c, v_pred = self._ppo_critic_forward(f)
        d_v = self.ppo_c_value * 2.0 * (v_pred - ret)

        g_W2c = d_v * h_c[np.newaxis, :]
        g_b2c = np.array([d_v])
        g_hc  = d_v * self.ppo_critic_W2.squeeze(0) * (h_cp > 0).astype(float)
        g_W1c = g_hc[:, np.newaxis] * f[np.newaxis, :]
        g_b1c = g_hc

        self.ppo_critic_W1 -= self.ppo_lr * g_W1c
        self.ppo_critic_b1 -= self.ppo_lr * g_b1c
        self.ppo_critic_W2 -= self.ppo_lr * g_W2c
        self.ppo_critic_b2 -= self.ppo_lr * g_b2c

    # ------------------------------------------------------------------
    # Unified update entry point (called from HeuristicRLAgent)
    # ------------------------------------------------------------------

    def record_outcome(self, seg_features: list[float], reward: float) -> None:
        """Update weights given the observed transition outcome for one segment."""
        if self.priority_mode == "vanilla_rl":
            self._update_vanilla_rl(seg_features, reward)
        elif self.priority_mode == "maml":
            self._update_maml_inner(seg_features, reward)
        elif self.priority_mode == "nn":
            self._update_nn(seg_features, reward)
        elif self.priority_mode == "sac":
            self._update_sac(seg_features, reward)
        elif self.priority_mode == "ppo":
            self._update_ppo(seg_features, reward)
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
                        prev_features[prev_action], reward
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
    """HeuristicRLAgent using Soft Actor-Critic for segment priority.

    Key differences from vanilla_rl / nn:
      - Twin Q-critics (reduces Q-value overestimation)
      - Off-policy replay buffer (better sample efficiency)
      - Entropy regularisation with learned temperature α (exploration)
      - Soft target network updates (training stability)
    """
    PRIORITY_MODE = "sac"


class HeuristicRLPPOAgent(HeuristicRLAgent):
    """HeuristicRLAgent using Proximal Policy Optimisation for segment priority.

    Key differences from the other RL agents:
      - On-policy rollout buffer (size 32) with full GAE-λ advantage estimates
      - Clipped surrogate objective (ε=0.2) prevents destructively large policy updates
      - Dedicated critic network provides a lower-variance value baseline
      - K=4 gradient epochs per rollout reuse collected data without going off-policy
      - Entropy bonus (c_e=0.01) maintains exploration throughout training
    """
    PRIORITY_MODE = "ppo"