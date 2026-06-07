# Learning Priority Functions for Graph-Based Exploration in ARC-AGI-3

**CS224R Final Project** — Ryan Bookman, Steve Mendeleev, Kyle Feinstein (Stanford University, 2026)

This repo extends the [graph-based exploration approach by Rudakov et al.](https://arxiv.org/abs/2512.24156) for ARC-AGI-3 by replacing its hand-coded priority heuristic with learned priority functions trained online via reinforcement learning.

## Overview

ARC-AGI-3 is an interactive benchmark where agents must discover unknown game mechanics through exploration under a strict action budget. The Rudakov et al. baseline tracks visited states in a directed graph and uses a fixed heuristic to assign each visual segment to one of five click-priority groups. This heuristic cannot adapt across levels.

We ask: can a *learned* priority function improve exploration efficiency across ARC-AGI-3's progressively harder levels?

### Key contributions

- Five online-learning agents that replace the fixed heuristic with a learned scoring function, all built on top of the existing graph explorer
- A richer four-feature segment representation (vs. three binary features in the original) enabling gradient-based learning
- Warm-started weights calibrated to recover the heuristic's prior ordering
- Online training from a dense ±1 binary reward signal (did clicking a segment cause a visual state transition?)

### Agents

| Agent | Approach |
|-------|----------|
| REINFORCE | Vanilla policy gradient — simple and interpretable |
| FOMAML | First-order MAML — meta-learns an initialization for fast per-level adaptation |
| Neural Net (MLP) | Two-layer MLP trained by online backpropagation |
| SAC | Soft Actor-Critic — off-policy with replay buffer, twin critics, entropy regularization |
| PPO | Proximal Policy Optimization — clipped surrogate objective with GAE-λ advantages |

### Results (45-min eval on ARC-AGI-3 `vc33`)

| Agent | Score | vs. Baseline (0.59) |
|-------|-------|---------------------|
| FOMAML | **4.42** | 7.5× |
| REINFORCE | 2.82 | 4.8× |
| SAC | 2.41 | 4.1× |
| PPO | 0.54 | ~baseline |
| Neural Net | 0.04 | below baseline |

FOMAML's dominance confirms that the primary bottleneck is fast per-level adaptation: different levels introduce distinct interactive elements, so an initialization trained for quick adaptation consistently outperforms a single accumulated weight vector.

## Quickstart

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if not already installed.

1. Clone this repo and enter the directory.

```bash
git clone https://github.com/StiopaPopa/224r-arc-agi-3.git
cd 224r-arc-agi-3
```

2. Copy `.env.example` to `.env` and set your ARC-AGI-3 API key.

```bash
cp .env.example .env
# edit .env and set ARC_API_KEY="your_api_key_here"
```

3. Run an agent. Available agent names: `heuristicagent`, `vanillaagent`, `mamlagent`, `nnagent`, `sacagent`, `ppoagent`.

```bash
uv run main.py --agent=mamlagent
```

To run across all games for a fixed duration (e.g. 45 minutes):

```bash
uv run batch_run.py --agent=mamlagent --minutes=45
```

## Paper

The full writeup is in [`CS224_Paper/cs224r_final_report_2026.pdf`](CS224_Paper/cs224r_final_report_2026.pdf).

## License

MIT License. See [LICENSE](LICENSE) for details.
