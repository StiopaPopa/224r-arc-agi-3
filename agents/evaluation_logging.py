from __future__ import annotations

import csv
import json
import logging
import math
import os
import statistics
import threading
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .structs import FrameData, GameAction, GameState, Scorecard

logger = logging.getLogger()


@dataclass
class EvaluationEvent:
    timestamp: str
    elapsed_seconds: float
    agent_name: str
    agent_class: str
    game_id: str
    action_index: int
    action_name: str
    action_value: int
    is_click: bool
    previous_score: int
    score: int
    score_delta: int
    previous_state: str
    state: str
    visual_transition: bool
    level_index: int
    solved_level: bool


class EvaluationLogger:
    """Non-invasive recorder for evaluation summaries and paper-ready plots."""

    def __init__(
        self,
        agent_name: str,
        *,
        root_dir: str = "results",
        interval_seconds: Optional[float] = None,
    ) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.agent_name = agent_name
        self.output_dir = Path(root_dir) / f"evaluation_{timestamp}"
        self.interval_seconds = interval_seconds or float(
            os.environ.get("EVALUATION_LOG_INTERVAL_SECONDS", "300")
        )
        self._lock = threading.Lock()
        self._events: list[EvaluationEvent] = []
        self._level_actions: dict[str, int] = defaultdict(int)
        self._level_index: dict[str, int] = defaultdict(int)
        self._solved_level_actions: list[dict[str, Any]] = []
        self._created = False
        self._finalized = False

    def _ensure_output_dir(self) -> None:
        if not self._created:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._created = True

    def record_action(
        self,
        *,
        agent: Any,
        action: GameAction,
        previous_frame: FrameData,
        frame: FrameData,
        action_index: int,
    ) -> None:
        """Observe an action after it has already been selected and executed."""

        try:
            elapsed_seconds = float(getattr(agent, "seconds", 0.0))
            game_id = getattr(agent, "game_id", frame.game_id)
            level_index = self._level_index[game_id]
            previous_score = int(previous_frame.score)
            score = int(frame.score)
            score_delta = score - previous_score
            solved_level = score_delta > 0 or frame.state == GameState.WIN
            visual_transition = _frames_differ(previous_frame, frame)
            is_click = action is GameAction.ACTION6

            if action is not GameAction.RESET:
                self._level_actions[game_id] += 1

            if solved_level:
                self._solved_level_actions.append(
                    {
                        "agent_name": self.agent_name,
                        "game_id": game_id,
                        "level_index": level_index,
                        "actions_to_solve": self._level_actions[game_id],
                        "score_after_solve": score,
                        "elapsed_seconds": elapsed_seconds,
                    }
                )
                self._level_index[game_id] = level_index + 1
                self._level_actions[game_id] = 0

            event = EvaluationEvent(
                timestamp=datetime.now().astimezone().isoformat(),
                elapsed_seconds=elapsed_seconds,
                agent_name=self.agent_name,
                agent_class=agent.__class__.__name__,
                game_id=game_id,
                action_index=action_index,
                action_name=action.name,
                action_value=int(action.value),
                is_click=is_click,
                previous_score=previous_score,
                score=score,
                score_delta=score_delta,
                previous_state=str(previous_frame.state.value),
                state=str(frame.state.value),
                visual_transition=visual_transition,
                level_index=level_index,
                solved_level=solved_level,
            )
        except Exception as exc:
            logger.warning(f"Evaluation logging skipped one action: {exc}")
            return

        with self._lock:
            self._events.append(event)

    def finalize(self, scorecard: Optional[Scorecard]) -> None:
        if self._finalized:
            return
        self._finalized = True
        with self._lock:
            events = list(self._events)
            solved_level_actions = list(self._solved_level_actions)

        self._ensure_output_dir()
        self._write_events(events)
        self._write_solved_levels(solved_level_actions)
        summary = self._build_summary(events, solved_level_actions, scorecard)
        time_series = self._build_time_series(events)
        transition_series = self._build_transition_series(events)
        self._write_dicts("time_series.csv", time_series)
        self._write_dicts("evaluation_summary.csv", summary)
        self._write_summary_markdown(summary)
        self._write_notes()
        self._write_scorecard(scorecard)
        self._write_plots(time_series, transition_series, summary, solved_level_actions)
        logger.info(f"Evaluation outputs saved to {self.output_dir}")

    def _write_events(self, events: list[EvaluationEvent]) -> None:
        rows = [asdict(event) for event in events]
        self._write_dicts("raw_action_log.csv", rows)
        with open(self.output_dir / "raw_action_log.json", "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)

    def _write_solved_levels(self, rows: list[dict[str, Any]]) -> None:
        self._write_dicts("actions_per_solved_level.csv", rows)

    def _write_dicts(self, filename: str, rows: list[dict[str, Any]]) -> None:
        path = self.output_dir / filename
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        fieldnames = list(rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _build_summary(
        self,
        events: list[EvaluationEvent],
        solved_level_actions: list[dict[str, Any]],
        scorecard: Optional[Scorecard],
    ) -> list[dict[str, Any]]:
        click_events = [e for e in events if e.is_click]
        transition_rate = _safe_ratio(
            sum(e.visual_transition for e in click_events), len(click_events)
        )
        first_25 = self._early_transition_rate(events, 25)
        first_50 = self._early_transition_rate(events, 50)
        actions_to_solve = [
            int(row["actions_to_solve"])
            for row in solved_level_actions
            if int(row["actions_to_solve"]) > 0
        ]

        final_score = scorecard.score if scorecard else sum(
            _latest_score_by_game(events).values()
        )
        levels_solved = scorecard.won if scorecard else len(solved_level_actions)
        total_actions = scorecard.total_actions if scorecard else sum(
            1 for e in events if e.action_name != "RESET"
        )

        return [
            {
                "agent_name": self.agent_name,
                "final_score": final_score,
                "levels_solved": levels_solved,
                "total_actions": total_actions,
                "transition_rate": _format_float(transition_rate),
                "average_actions_per_solved_level": _format_float(
                    _mean(actions_to_solve)
                ),
                "median_actions_per_solved_level": _format_float(
                    _median(actions_to_solve)
                ),
                "min_actions_per_solved_level": min(actions_to_solve)
                if actions_to_solve
                else "",
                "max_actions_per_solved_level": max(actions_to_solve)
                if actions_to_solve
                else "",
                "first_25_click_transition_rate": _format_float(first_25),
                "first_50_click_transition_rate": _format_float(first_50),
                "num_logged_actions": len(events),
                "num_logged_clicks": len(click_events),
            }
        ]

    def _build_time_series(
        self, events: list[EvaluationEvent]
    ) -> list[dict[str, Any]]:
        if not events:
            return []

        events = sorted(events, key=lambda e: e.elapsed_seconds)
        latest_scores: dict[str, int] = {}
        solved_levels: set[tuple[str, int]] = set()
        rows: list[dict[str, Any]] = []
        next_bucket = 0.0

        for event in events:
            latest_scores[event.game_id] = event.score
            if event.solved_level:
                solved_levels.add((event.game_id, event.level_index))
            while event.elapsed_seconds >= next_bucket:
                rows.append(
                    {
                        "elapsed_seconds": round(next_bucket, 3),
                        "agent_name": self.agent_name,
                        "score": sum(latest_scores.values()),
                        "levels_solved": len(solved_levels),
                    }
                )
                next_bucket += self.interval_seconds

        final_elapsed = max(event.elapsed_seconds for event in events)
        rows.append(
            {
                "elapsed_seconds": round(final_elapsed, 3),
                "agent_name": self.agent_name,
                "score": sum(latest_scores.values()),
                "levels_solved": len(solved_levels),
            }
        )
        return _dedupe_rows(rows, "elapsed_seconds")

    def _build_transition_series(
        self, events: list[EvaluationEvent]
    ) -> list[dict[str, Any]]:
        click_events = sorted(
            [event for event in events if event.is_click],
            key=lambda e: e.elapsed_seconds,
        )
        if not click_events:
            return []
        rows: list[dict[str, Any]] = []
        transitions = 0
        next_bucket = 0.0
        seen = 0
        for event in click_events:
            seen += 1
            transitions += int(event.visual_transition)
            while event.elapsed_seconds >= next_bucket:
                rows.append(
                    {
                        "elapsed_seconds": round(next_bucket, 3),
                        "agent_name": self.agent_name,
                        "transition_rate": _safe_ratio(transitions, seen),
                        "num_clicks": seen,
                    }
                )
                next_bucket += self.interval_seconds
        rows.append(
            {
                "elapsed_seconds": round(click_events[-1].elapsed_seconds, 3),
                "agent_name": self.agent_name,
                "transition_rate": _safe_ratio(transitions, seen),
                "num_clicks": seen,
            }
        )
        return _dedupe_rows(rows, "elapsed_seconds")

    def _early_transition_rate(
        self, events: list[EvaluationEvent], click_limit: int
    ) -> Optional[float]:
        by_level: dict[tuple[str, int], list[EvaluationEvent]] = defaultdict(list)
        for event in events:
            if event.is_click:
                by_level[(event.game_id, event.level_index)].append(event)
        rates = [
            _safe_ratio(sum(e.visual_transition for e in level_events[:click_limit]), len(level_events[:click_limit]))
            for level_events in by_level.values()
            if level_events[:click_limit]
        ]
        rates = [rate for rate in rates if rate is not None]
        return _mean(rates)

    def _write_summary_markdown(self, summary: list[dict[str, Any]]) -> None:
        path = self.output_dir / "evaluation_summary.md"
        if not summary:
            path.write_text("No evaluation events were logged.\n", encoding="utf-8")
            return
        headers = list(summary[0].keys())
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
        for row in summary:
            lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _write_notes(self) -> None:
        notes = [
            "# Evaluation Logging Notes",
            "",
            "- Instrumentation observes chosen actions and returned frames only; it does not select actions, mutate frames, or update training state.",
            "- Level boundaries are inferred from score increases because the API frame schema does not expose an explicit level id.",
            "- Transition rate is computed for click actions (`ACTION6`) as visual frame changes divided by total logged clicks.",
            "- Early adaptation rates average each level's first 25 and first 50 logged clicks. Levels with no clicks are skipped for those rates.",
            "- Multiple-run aggregation is not performed here because the current runner executes one scorecard per invocation; CSV outputs are shaped for later aggregation.",
        ]
        (self.output_dir / "notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")

    def _write_scorecard(self, scorecard: Optional[Scorecard]) -> None:
        if scorecard is None:
            return
        with open(self.output_dir / "scorecard.json", "w", encoding="utf-8") as f:
            json.dump(scorecard.model_dump(), f, indent=2, default=str)

    def _write_plots(
        self,
        time_series: list[dict[str, Any]],
        transition_series: list[dict[str, Any]],
        summary: list[dict[str, Any]],
        solved_level_actions: list[dict[str, Any]],
    ) -> None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            logger.warning(f"Skipping evaluation plots; matplotlib unavailable: {exc}")
            with open(self.output_dir / "plotting_skipped.txt", "w", encoding="utf-8") as f:
                f.write(f"matplotlib unavailable: {exc}\n")
            return

        plt.rcParams.update(
            {
                "figure.dpi": 140,
                "savefig.dpi": 300,
                "font.size": 11,
                "axes.titlesize": 13,
                "axes.labelsize": 11,
                "legend.fontsize": 10,
            }
        )
        self._plot_line(
            plt,
            time_series,
            y_key="score",
            ylabel="Cumulative score",
            title="Score Over Time",
            filename="score_over_time.png",
        )
        self._plot_line(
            plt,
            time_series,
            y_key="levels_solved",
            ylabel="Levels solved",
            title="Levels Solved Over Time",
            filename="levels_solved_over_time.png",
        )
        self._plot_line(
            plt,
            transition_series,
            y_key="transition_rate",
            ylabel="Click transition rate",
            title="Transition Rate Over Time",
            filename="transition_rate_over_time.png",
        )
        self._plot_actions_per_level(plt, solved_level_actions)
        self._plot_summary_table(plt, summary)
        self._plot_main_figure(plt, time_series, transition_series, summary)

    def _plot_line(
        self,
        plt: Any,
        rows: list[dict[str, Any]],
        *,
        y_key: str,
        ylabel: str,
        title: str,
        filename: str,
    ) -> None:
        if not rows:
            return
        x = [float(row["elapsed_seconds"]) / 60.0 for row in rows]
        y = [float(row[y_key]) for row in rows]
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        ax.plot(x, y, marker="o", linewidth=2.2, markersize=4, label=self.agent_name)
        ax.set_title(title)
        ax.set_xlabel("Elapsed time (minutes)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(self.output_dir / filename)
        plt.close(fig)

    def _plot_actions_per_level(
        self, plt: Any, solved_level_actions: list[dict[str, Any]]
    ) -> None:
        if not solved_level_actions:
            return
        labels = [
            f"{row['game_id']} L{row['level_index']}"
            for row in solved_level_actions
        ]
        values = [int(row["actions_to_solve"]) for row in solved_level_actions]
        fig_width = max(7.2, min(14.0, 0.45 * len(values)))
        fig, ax = plt.subplots(figsize=(fig_width, 4.4))
        ax.bar(range(len(values)), values, color="#4C78A8")
        ax.set_title("Actions Per Solved Level")
        ax.set_xlabel("Solved level")
        ax.set_ylabel("Actions to solve")
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(self.output_dir / "actions_per_solved_level.png")
        plt.close(fig)

    def _plot_summary_table(self, plt: Any, summary: list[dict[str, Any]]) -> None:
        if not summary:
            return
        compact_headers = [
            "agent_name",
            "final_score",
            "levels_solved",
            "total_actions",
            "transition_rate",
            "average_actions_per_solved_level",
            "first_25_click_transition_rate",
            "first_50_click_transition_rate",
        ]
        row = summary[0]
        values = [[str(row.get(header, "")) for header in compact_headers]]
        fig, ax = plt.subplots(figsize=(12, 2.2))
        ax.axis("off")
        table = ax.table(
            cellText=values,
            colLabels=compact_headers,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.35)
        fig.tight_layout()
        fig.savefig(self.output_dir / "evaluation_summary_table.png")
        plt.close(fig)

    def _plot_main_figure(
        self,
        plt: Any,
        time_series: list[dict[str, Any]],
        transition_series: list[dict[str, Any]],
        summary: list[dict[str, Any]],
    ) -> None:
        if not time_series:
            return
        fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
        x = [float(row["elapsed_seconds"]) / 60.0 for row in time_series]
        axes[0, 0].plot(x, [float(row["score"]) for row in time_series], color="#4C78A8", linewidth=2.2)
        axes[0, 0].set_title("Score over time")
        axes[0, 0].set_xlabel("Minutes")
        axes[0, 0].set_ylabel("Score")

        axes[0, 1].plot(x, [float(row["levels_solved"]) for row in time_series], color="#59A14F", linewidth=2.2)
        axes[0, 1].set_title("Levels solved over time")
        axes[0, 1].set_xlabel("Minutes")
        axes[0, 1].set_ylabel("Levels solved")

        if transition_series:
            tx = [float(row["elapsed_seconds"]) / 60.0 for row in transition_series]
            axes[1, 0].plot(tx, [float(row["transition_rate"]) for row in transition_series], color="#F28E2B", linewidth=2.2)
        axes[1, 0].set_title("Click transition rate over time")
        axes[1, 0].set_xlabel("Minutes")
        axes[1, 0].set_ylabel("Transition rate")
        axes[1, 0].set_ylim(0, 1)

        axes[1, 1].axis("off")
        if summary:
            row = summary[0]
            lines = [
                f"Final score: {row['final_score']}",
                f"Levels solved: {row['levels_solved']}",
                f"Total actions: {row['total_actions']}",
                f"Transition rate: {row['transition_rate']}",
                f"Avg actions/solved level: {row['average_actions_per_solved_level']}",
                f"First-25 click transition rate: {row['first_25_click_transition_rate']}",
                f"First-50 click transition rate: {row['first_50_click_transition_rate']}",
            ]
            axes[1, 1].text(0.02, 0.95, "\n".join(lines), va="top", fontsize=12)
        axes[1, 1].set_title("Final summary")

        for ax in axes.ravel()[:3]:
            ax.grid(True, alpha=0.25)
        fig.suptitle(f"Evaluation Summary: {self.agent_name}", fontsize=15)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(self.output_dir / "main_evaluation_figure.png")
        plt.close(fig)


def _frames_differ(previous_frame: FrameData, frame: FrameData) -> bool:
    return previous_frame.frame != frame.frame or previous_frame.state != frame.state


def _safe_ratio(numerator: float, denominator: int) -> Optional[float]:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def _mean(values: list[float] | list[int]) -> Optional[float]:
    if not values:
        return None
    return float(statistics.mean(values))


def _median(values: list[int]) -> Optional[float]:
    if not values:
        return None
    return float(statistics.median(values))


def _format_float(value: Optional[float]) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return f"{value:.4f}"


def _latest_score_by_game(events: list[EvaluationEvent]) -> dict[str, int]:
    scores: dict[str, int] = {}
    for event in sorted(events, key=lambda e: e.elapsed_seconds):
        scores[event.game_id] = event.score
    return scores


def _dedupe_rows(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    deduped: dict[Any, dict[str, Any]] = {}
    for row in rows:
        deduped[row[key]] = row
    return list(deduped.values())
