import numpy as np
import pytest

from agents import AVAILABLE_AGENTS
from agents.heuristic_rl_prioritiy_agent import FrameProcessorRL, HeuristicRLSACAgent


@pytest.mark.unit
class TestSACPriorityProcessor:
    def test_sac_agent_is_available(self):
        assert AVAILABLE_AGENTS["heuristicrlsacagent"] is HeuristicRLSACAgent

    def test_sac_priority_groups_cover_each_segment_once(self):
        processor = FrameProcessorRL(priority_mode="sac")
        features = [
            [True, True, False],
            [False, True, False],
            [False, False, True],
            [True, False, False],
            [False, False, False],
            [True, True, False],
        ]

        groups = processor.create_priority_groups(features)

        assert len(groups) == 5
        grouped_ids = set().union(*groups)
        assert grouped_ids == set(range(len(features)))
        assert sum(len(group) for group in groups) == len(features)

    def test_sac_uses_richer_segment_features_without_affecting_other_modes(self):
        segments = [
            {
                "bounding_box": (2, 2, 8, 9),
                "color": 8,
                "area": 56,
                "is_rectangle": True,
                "number_of_twins": 0,
            },
            {
                "bounding_box": (0, 0, 63, 1),
                "color": 16,
                "area": 128,
                "is_rectangle": True,
                "number_of_twins": 0,
            },
        ]

        sac_processor = FrameProcessorRL(priority_mode="sac")
        sac_processor.frame_segments_to_action_groups(segments, 5)

        heuristic_processor = FrameProcessorRL(priority_mode="heuristic")
        heuristic_processor.frame_segments_to_action_groups(segments, 5)

        assert len(sac_processor.last_features_list[0]) == 10
        assert len(heuristic_processor.last_features_list[0]) == 3

    def test_sac_update_changes_actor_and_critics(self):
        processor = FrameProcessorRL(priority_mode="sac")
        frame_features = [
            [True, True, False],
            [False, True, False],
            [False, False, True],
        ]

        actor_before = processor.sac_actor_weights.copy()
        q1_before = processor.sac_q1_weights.copy()
        q2_before = processor.sac_q2_weights.copy()

        processor.record_outcome(
            frame_features[0],
            reward=1.0,
            frame_features=frame_features,
            action_index=0,
        )

        assert processor.sac_update_count == 1
        assert not np.allclose(processor.sac_actor_weights, actor_before)
        assert not np.allclose(processor.sac_q1_weights, q1_before)
        assert not np.allclose(processor.sac_q2_weights, q2_before)

    def test_sac_positive_reward_increases_selected_q_values(self):
        processor = FrameProcessorRL(priority_mode="sac")
        feature = [True, True, False]
        augmented = processor._augment_features(feature)[0]

        q1_before = float(augmented @ processor.sac_q1_weights)
        q2_before = float(augmented @ processor.sac_q2_weights)

        processor.record_outcome(feature, reward=1.0)

        assert float(augmented @ processor.sac_q1_weights) > q1_before
        assert float(augmented @ processor.sac_q2_weights) > q2_before
