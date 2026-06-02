import csv
from pathlib import Path

from agents.evaluation_logging import EvaluationLogger
from agents.structs import FrameData, GameAction, GameState


class DummyAgent:
    game_id = "game-a"
    seconds = 0.0


def test_evaluation_logger_writes_summary_and_raw_logs(tmp_path: Path):
    logger = EvaluationLogger(
        "test-agent",
        root_dir=str(tmp_path),
        interval_seconds=1,
    )
    agent = DummyAgent()

    first = FrameData(
        game_id="game-a",
        frame=[[[0, 0], [0, 0]]],
        state=GameState.NOT_FINISHED,
        score=0,
    )
    second = FrameData(
        game_id="game-a",
        frame=[[[1, 0], [0, 0]]],
        state=GameState.NOT_FINISHED,
        score=0,
    )
    solved = FrameData(
        game_id="game-a",
        frame=[[[1, 1], [0, 0]]],
        state=GameState.WIN,
        score=1,
    )

    agent.seconds = 0.5
    logger.record_action(
        agent=agent,
        action=GameAction.ACTION6,
        previous_frame=first,
        frame=second,
        action_index=0,
    )
    agent.seconds = 1.5
    logger.record_action(
        agent=agent,
        action=GameAction.ACTION6,
        previous_frame=second,
        frame=solved,
        action_index=1,
    )

    logger.finalize(None)

    output_dirs = list(tmp_path.glob("evaluation_*"))
    assert len(output_dirs) == 1
    output_dir = output_dirs[0]
    assert (output_dir / "raw_action_log.csv").exists()
    assert (output_dir / "raw_action_log.json").exists()
    assert (output_dir / "evaluation_summary.csv").exists()
    assert (output_dir / "evaluation_summary.md").exists()
    assert (output_dir / "actions_per_solved_level.csv").exists()
    assert (output_dir / "time_series.csv").exists()
    assert (output_dir / "notes.md").exists()

    with open(output_dir / "evaluation_summary.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert rows[0]["agent_name"] == "test-agent"
    assert rows[0]["final_score"] == "1"
    assert rows[0]["levels_solved"] == "1"
    assert rows[0]["total_actions"] == "2"
    assert rows[0]["transition_rate"] == "1.0000"
    assert rows[0]["average_actions_per_solved_level"] == "2.0000"
    assert rows[0]["first_25_click_transition_rate"] == "1.0000"
