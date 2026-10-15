# MazePilot RL

**Watch a car learn to explore branching mazes using tabular Q-learning.**

MazePilot RL is a small educational reinforcement-learning project built with
Python and Tkinter. A car explores randomly generated 2D mazes using only the
cells immediately ahead, left, and right, plus a small amount of movement memory.
Watch the saved policy navigate new maze seeds, inspect its action scores, or
replay ten trials spanning an entire training run.

No GPU, neural network, Gymnasium, or third-party Python packages are required.

## Features

- Small mazes with junctions, dead ends, and occasional loops.
- Three actions: move forward, turn left, and turn right.
- Local sensing, visit-count memory, and a readable saved Q-table.
- Adjustable-speed demo with pause, single-step, and new-maze controls.
- Ten-panel playback of real training trials, including early failures.
- Training progress, saved policies, evaluation, and a random-action baseline.
- Tests for connectivity, sensor isolation, backtracking, and recording integrity.

## Quick start

Install Python 3.10 or newer with Tkinter. Download or clone this repository,
then open a terminal in the folder containing `maze_car.py`.

Check whether Tkinter is available:

```bash
python3 -c "import tkinter; print(tkinter.TkVersion)"
```

Start the interactive demo with the included trained policy:

```bash
python3 maze_car.py demo
```

On Windows, use `python` or `py` instead of `python3` if appropriate.
If Tkinter is missing, install Tk support for the Python distribution you use;
Tkinter is not installed through this project's requirements file. A desktop
display is required for the GUI; training and evaluation can run without one.

In PyCharm, open the **whole repository folder** and run `maze_car.py` or
`learning_view.py`. Copying just the viewer to another folder will break its
imports of `maze_car.py` and `ui_style.py`.

### Demo controls

| Control | Purpose |
| --- | --- |
| Play / pause | Run or pause the loaded policy |
| Single step | Execute one decision and inspect the action scores |
| New maze | Reset movement memory and generate a different maze seed |
| Train + save | Train from scratch and replace the saved policy |
| Show full maze | Reveal the map to the human viewer only |
| Action delay | Set playback delay from 200 to 1,000 ms |

Orange triangle = car; green cell = goal; gold outline = local view;
turquoise cells = visited. Full-map display never changes the policy's input.
The initial playback delay is 450 ms per action.

### Watch the learning progression

```bash
python3 learning_view.py
```

Ten panels replay actual trials sampled from episode 1 through episode 30,000,
with a different maze seed in each panel. Each shows collisions, reward, and
the selected action's Q-score update. A graph summarizes training success in
100-episode blocks.

These are recordings replayed **after training**, not ten independent learners
or all episodes rendered live. The default replay delay is 35 ms per action.
Use Replay all 10, Pause / resume, or Train fresh + record. Different maze
seeds and exploration rates mean individual panels are illustrative rather
than a controlled comparison of policy quality.

## Train and evaluate

```bash
python3 maze_car.py train --episodes 30000
python3 maze_car.py evaluate
python3 compare.py
python3 -m unittest -v
```

Training starts from an empty Q-table. It replaces `outputs/policy.json`,
`outputs/training_timeline.json`, and `outputs/training.csv`.
Copy these files elsewhere first if you want to keep an earlier run.
After retraining, rerun `compare.py` to refresh the comparison report.
Restart an already-open demo to load an externally retrained policy.

Outputs are located relative to the source file, regardless of the terminal's
working directory. The bundled policy and recordings let you start immediately.

## How the policy works

The policy receives:

```text
(left cell, front cell, right cell, consecutive turns)
```

| Cell code | Meaning |
| --- | --- |
| 0 | Wall |
| 1 | Open, never visited |
| 2 | Visited once |
| 3 | Visited twice |
| 4 | Visited three or more times |
| 5 | Visible adjacent goal |

The turn counter is capped at two. Successful forward movement resets it;
turning increases it and a collision leaves it unchanged. Visit memory resets
for each maze. Both are explicitly programmed memory, not learned localization.
The simulation assumes exact knowledge of the car's own grid position to
maintain that memory.

Each observation maps to three learned scores:

```text
Q[observation] = [forward score, left-turn score, right-turn score]
```

During training, exploratory actions provide experience for the update:

```text
Q(o, a) ← Q(o, a) + α × [r + γ × max Q(o_next, ·) − Q(o, a)]
```

The future-value term is zero at episode end. Here α = 0.12 and γ = 0.97.
Exploration decreases from 100% to 4%. During playback, the car selects a
highest-scoring action with random tie-breaking; it does not update the Q-table.

| Outcome | Reward |
| --- | ---: |
| Reach goal | +10 |
| Enter an unvisited cell | +0.30 |
| Revisit a cell | −0.04 |
| Turn 90 degrees | −0.20 |
| Hit wall | −1 |

Revisiting has a small penalty so backtracking remains possible. Turning costs
more to discourage unnecessary rotations. Episodes stop at the goal or after
400 actions. The agent has no hidden-goal bearing, global maze input, or route
planner. Graph operations are used only to generate the environment.

## Included benchmark

The bundled [comparison report](outputs/comparison.json) evaluates both policies
on 500 identical maze seeds starting at 4,000,000.

| Metric | Random actions | Learned policy |
| --- | ---: | ---: |
| Goal success | 41.6% | 100% |
| Mean actions per episode | 321.612 | 23.160 |
| Mean turns per episode | 214.892 | 5.920 |
| Total wall collisions | 31,963 | 0 |
| Mean consecutive opposite-turn pairs | 71.726 | 0 |

Action averages include failed episodes, which stop at the time limit.
These results describe the included run; they are not a guarantee for arbitrary
mazes or different training settings.

## Scope and limitations

- This is a grid-world learning demonstration, not a physical vehicle simulator.
- Mazes use a connected 3×3 node graph rendered on a 9×9 grid, with 17 or 18
  open cells. Larger mazes are not evaluated.
- Training uses seeds 0–29,999 by default. Evaluation uses separate seed ranges,
  but different seeds can generate the same geometry. This is not a
  geometry-disjoint generalization benchmark.
- Local observations and limited memory make the problem partially observed;
  ordinary tabular Q-learning has no general optimality guarantee here.
- Exploration rewards encourage finding new cells, not necessarily the shortest
  route. Random actions are a basic baseline, not an optimal planner.
- An action score estimates discounted future reward; it is not a probability.

## Repository layout

```text
maze_car.py                  Environment, Q-learning, CLI, and single-maze demo
learning_view.py             Ten-panel training replay
ui_style.py                  Shared GUI styles
compare.py                   Random-versus-learned evaluation
test_maze.py                 Automated checks
outputs/
  policy.json                Saved Q-table and training metadata
  training_timeline.json     Recorded trials and learning history
  training.csv               Training progress
  comparison.json            Evaluation comparison
requirements.txt             Documents standard-library-only dependencies
LICENSE                      MIT license
```

## License

[MIT](LICENSE). Copyright (c) 2026 Alex Chen.
