"""Branching mazes with Q-learning, local sensors and visit-count memory. Python 3.10+."""
import argparse
import csv
import json
import random
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent
POLICY = ROOT / "outputs" / "policy.json"
DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
ACTIONS = ["forward", "turn left", "turn right"]


def generate(seed):
    """Connected 3x3-node mazes, guaranteed junctions and dead ends, optional loop."""
    rng = random.Random(seed)
    nodes = [(x, y) for x in range(3) for y in range(3)]
    edges = [(p, (p[0]+dx, p[1]+dy)) for p in nodes
             for dx, dy in [(1, 0), (0, 1)]
             if p[0]+dx < 3 and p[1]+dy < 3]
    while True:
        parent = {p: p for p in nodes}
        def find(p):
            while parent[p] != p:
                p = parent[p]
            return p
        shuffled = edges[:]
        rng.shuffle(shuffled)
        tree = []
        for a, b in shuffled:
            if find(a) != find(b):
                parent[find(a)] = find(b)
                tree.append((a, b))
        def degrees(es):
            return {p: sum(p in e for e in es) for p in nodes}
        deg = degrees(tree)
        if max(deg.values()) >= 3 and sum(v == 1 for v in deg.values()) >= 3:
            break
    if seed % 3 == 0:
        candidates = [e for e in edges if e not in tree
                      and sum(v == 1 for v in degrees(tree+[e]).values()) >= 2]
        if candidates:
            tree.append(rng.choice(candidates))
    deg = degrees(tree)
    leaves = [p for p in nodes if deg[p] == 1]
    start = rng.choice(leaves)
    distance, pending = {start: 0}, [start]
    while pending:
        p = pending.pop(0)
        for edge in tree:
            if p in edge:
                other = edge[1] if edge[0] == p else edge[0]
                if other not in distance:
                    distance[other] = distance[p]+1
                    pending.append(other)
    choices = [p for p in leaves if distance[p] >= 3]
    goal = rng.choice(choices) if choices else max(leaves, key=distance.get)
    def cell(p):
        return (2*p[0]+2, 2*p[1]+2)
    floor = {cell(p) for p in nodes}
    for a, b in tree:
        ax, ay = cell(a)
        bx, by = cell(b)
        floor.add(((ax+bx)//2, (ay+by)//2))
    pos = cell(start)
    heading = next(i for i, (dx, dy) in enumerate(DIRS)
                   if (pos[0]+dx, pos[1]+dy) in floor)
    return floor, pos, cell(goal), heading


class Maze:
    def __init__(self, seed):
        self.seed = seed
        self.floor, self.pos, self.goal, self.heading = generate(seed)
        self.start = self.pos
        self.visited = {self.pos}
        self.visit_counts = {self.pos: 1}
        self.turn_streak = 0
        self.backtracks = 0
        self.steps = self.collisions = 0
        self.done = self.success = False

    def nearby(self):
        # Immediate left/front/right cells only. No rear or global goal bearing.
        result = []
        for rotation in (-1, 0, 1):
            dx, dy = DIRS[(self.heading + rotation) % 4]
            result.append((self.pos[0] + dx, self.pos[1] + dy))
        return result

    def observation(self):
        # Local sensor plus remembered visits. No hidden goal bearing or global map.
        local = tuple(0 if p not in self.floor else 5 if p == self.goal
                      else 1 + min(3, self.visit_counts.get(p, 0))
                      for p in self.nearby())
        return local + (min(2, self.turn_streak),)

    def step(self, action):
        if self.done:
            raise RuntimeError("Reset the maze before stepping a finished episode")
        self.steps += 1
        reward = -0.20
        if action == 0:
            target = self.nearby()[1]
            if target not in self.floor:
                self.collisions += 1
                reward = -1.0
            else:
                reward = -0.04 if target in self.visited else 0.3
                self.backtracks += target in self.visited
                self.visit_counts[target] = self.visit_counts.get(target, 0) + 1
                self.turn_streak = 0
                self.pos = target
                self.visited.add(target)
        else:
            self.turn_streak += 1
            self.heading = (self.heading + (-1 if action == 1 else 1)) % 4
        self.success = self.pos == self.goal
        if self.success:
            reward = 10.0
        self.done = self.success or self.steps >= 400
        return self.observation(), reward, self.done


def key(obs):
    return ",".join(map(str, obs))


def values(q, obs):
    return q.setdefault(key(obs), [0.0, 0.0, 0.0])


def choose(q, obs, rng, epsilon=0.0):
    if rng.random() < epsilon:
        return rng.randrange(3)
    scores = q.get(key(obs), [0.0, 0.0, 0.0])
    best = max(scores)
    return rng.choice([i for i, value in enumerate(scores) if value == best])


def evaluate(q, episodes=200, seed=1_000_000):
    rng = random.Random(12345)
    success = steps = collisions = turns = opposite_pairs = 0
    for i in range(episodes):
        env = Maze(seed + i)
        previous = None
        while not env.done:
            action = choose(q, env.observation(), rng)
            turns += action != 0
            opposite_pairs += previous in (1, 2) and action in (1, 2) and previous != action
            env.step(action)
            previous = action
        success += env.success
        steps += env.steps
        collisions += env.collisions
    return {"episodes": episodes, "success_rate": success / episodes,
            "mean_steps": steps / episodes, "collisions": collisions,
            "mean_turns": turns / episodes,
            "mean_opposite_turn_pairs": opposite_pairs / episodes}


def train(episodes=30000, seed=42, progress=None):
    rng, q, history = random.Random(seed), {}, []
    selected = {round(i * (episodes - 1) / 9) for i in range(10)}
    traces = []
    wins = 0
    for episode in range(episodes):
        # Training seeds never overlap the default held-out evaluation seeds.
        env = Maze(episode)
        epsilon = max(0.04, 1.0 - episode / max(1, episodes * 0.8))
        obs = env.observation()
        trace = None
        if episode in selected:
            trace = {"episode": episode + 1, "maze_seed": episode,
                     "epsilon": epsilon, "frames": [{"pos": env.pos,
                     "heading": env.heading, "action": "start", "reward": 0,
                     "collisions": 0, "q_before": 0, "q_after": 0}]}
        while not env.done:
            action = choose(q, obs, rng, epsilon)
            next_obs, reward, done = env.step(action)
            old = values(q, obs)
            before = old[action]
            target = reward + (0 if done else 0.97 * max(values(q, next_obs)))
            old[action] += 0.12 * (target - old[action])
            if trace is not None:
                trace["frames"].append({"pos": env.pos, "heading": env.heading,
                    "action": ACTIONS[action], "reward": reward,
                    "collisions": env.collisions, "q_before": before,
                    "q_after": old[action]})
            obs = next_obs
        if trace is not None:
            trace["success"] = env.success
            traces.append(trace)
        wins += env.success
        if (episode + 1) % 100 == 0 or episode == episodes - 1:
            block = 100 if (episode + 1) % 100 == 0 else (episode + 1) % 100
            row = {"episode": episode + 1, "epsilon": epsilon,
                   "training_success_rate": wins / block}
            history.append(row)
            if progress:
                progress(row)
            wins = 0
    report = evaluate(q)
    POLICY.parent.mkdir(exist_ok=True)
    payload = {"format_version": 2, "algorithm": "tabular Q-learning",
               "training_episodes": episodes, "seed": seed,
               "observations": "left,front,right: wall=0,new=1,visits1=2,visits2=3,visits3plus=4,goal=5; turn_streak capped at 2",
               "actions": ACTIONS, "evaluation": report, "q_table": q}
    temporary = POLICY.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(POLICY)
    timeline = POLICY.parent / "training_timeline.json"
    temp_timeline = timeline.with_suffix(".tmp")
    temp_timeline.write_text(json.dumps({"episodes": episodes, "seed": seed,
                              "traces": traces, "history": history}))
    temp_timeline.replace(timeline)
    with (POLICY.parent / "training.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    return q, report


def load():
    data = json.loads(POLICY.read_text())
    if data.get("format_version") != 2:
        raise ValueError("Unsupported policy format")
    return data["q_table"]


def demo():
    import tkinter as tk
    from tkinter import ttk
    from ui_style import apply_style
    import queue
    root = tk.Tk()
    root.title("Branching Maze Car — learn, explore, backtrack")
    apply_style(root)
    root.resizable(False, False)
    q = load() if POLICY.exists() else {}
    events = queue.Queue()
    rng = random.Random()
    state = {"q": q, "playing": False, "training": False, "env": None}
    ttk.Label(root, text="BRANCHING MAZE CAR · local vision + memory",
              font=("Helvetica", 19, "bold")).pack(padx=20, pady=(16, 5))
    ttk.Label(root, text="Junctions, dead ends and loops • local sensors • no full-map input").pack()
    controls = ttk.Frame(root)
    controls.pack(pady=12)
    canvas = tk.Canvas(root, width=680, height=455, bg="#101827", highlightthickness=0)
    canvas.pack(padx=12)
    status = tk.StringVar(value="Saved policy loaded" if q else "Click Train to learn a policy first")
    ttk.Label(root, textvariable=status, wraplength=650).pack(padx=20, pady=10)
    reveal = tk.BooleanVar(value=True)
    ttk.Checkbutton(root, text="Show full maze to human viewer (never to the policy)",
                    variable=reveal).pack()
    delay = tk.IntVar(value=450)
    ttk.Label(root, text="Action delay (milliseconds): slower ← adjust → faster is lower").pack(pady=(8, 0))
    tk.Scale(root, from_=200, to=1000, orient="horizontal", variable=delay,
             length=400).pack(pady=(0, 12))

    def new_maze():
        state["playing"] = False
        state["env"] = Maze(rng.randrange(2_000_000, 2_000_000_000))
        status.set("New maze ready. Press Play or Single step.")
        draw()

    def draw():
        canvas.delete("all")
        env = state["env"]
        visible = set(env.nearby()) | {env.pos}
        for y in range(9):
            for x in range(9):
                p = (x, y)
                known = reveal.get() or p in visible or p in env.visited
                color = "#26344a" if p not in env.floor else "#e4edf4"
                if not known:
                    color = "#111a2b"
                elif p in env.visited:
                    color = "#a2d6ca"
                if known and p == env.goal:
                    color = "#3acf83"
                if p in visible:
                    outline = "#f5bd4f"
                else:
                    outline = "#101827"
                canvas.create_rectangle(10+x*47, 10+y*47, 56+x*47, 56+y*47,
                                        fill=color, outline=outline, width=2)
        cx, cy = 33.5+env.pos[0]*47, 33.5+env.pos[1]*47
        dx, dy = DIRS[env.heading]
        canvas.create_polygon(cx+dx*18, cy+dy*18, cx-dx*12-dy*11,
                              cy-dy*12+dx*11, cx-dx*12+dy*11, cy-dy*12-dx*11,
                              fill="#ef6a48", outline="#ffffff", width=2)
        canvas.create_text(565, 27, text="WHAT THE CAR SEES", fill="white", font=("Helvetica", 13, "bold"))
        labels = ["WALL", "NEW", "VISITED ONCE", "VISITED TWICE", "VISITED 3+", "GOAL"]
        for i, (name, value) in enumerate(zip(("Left edge", "Ahead", "Right edge"), env.observation())):
            canvas.create_text(465, 70+i*35, anchor="w", text=f"{name}: {labels[value]}", fill="#e4edf4")
        canvas.create_text(465, 170, anchor="w", text=f"Turn memory: {min(2, env.turn_streak)}", fill="#acbbd1")
        canvas.create_text(465, 190, anchor="w", text="Learned action scores (Q)", fill="white")
        for i, (name, score) in enumerate(zip(ACTIONS, state["q"].get(key(env.observation()), [0, 0, 0]))):
            canvas.create_text(465, 220+i*27, anchor="w", text=f"{name}: {score:.2f}", fill="#91d8cc")
        canvas.create_text(465, 330, anchor="w", text=f"Actions: {env.steps} / 400\nCollisions: {env.collisions}\nRevisit moves: {env.backtracks}\nGreen = goal · gold = view", fill="#c4cede")

    def step():
        env = state["env"]
        if state["training"] or env.done:
            return
        if not state["q"]:
            status.set("Train a policy first.")
            return
        action = choose(state["q"], env.observation(), rng)
        _, reward, _ = env.step(action)
        status.set(f"{ACTIONS[action]} · reward {reward:+.2f} · maze seed {env.seed}")
        if env.done:
            state["playing"] = False
            status.set("Goal reached! Generate a new maze to test again." if env.success else "Time limit reached. Generate a new maze or retrain.")
        draw()

    def toggle():
        if not state["training"]:
            state["playing"] = not state["playing"]

    def start_training():
        if state["training"]:
            return
        state["playing"] = False
        state["training"] = True
        status.set("Training on 30,000 generated maze seeds…")
        def work():
            try:
                result = train(progress=lambda row: events.put(("progress", row)))
                events.put(("done", result))
            except Exception as error:
                events.put(("error", str(error)))
        threading.Thread(target=work, daemon=True).start()

    for label, command in [("Train + save", start_training), ("New maze", new_maze),
                            ("Play / pause", toggle), ("Single step", step)]:
        ttk.Button(controls, text=label, command=command).pack(side="left", padx=4)

    def tick():
        while not events.empty():
            kind, data = events.get_nowait()
            if kind == "progress":
                status.set(f"Training {data['episode']}/30000 · recent success {data['training_success_rate']:.0%} · exploration {data['epsilon']:.0%}")
            else:
                state["training"] = False
                if kind == "done":
                    state["q"], report = data
                    status.set(f"Saved policy · held-out seed success: {report['success_rate']:.1%} over {report['episodes']} mazes")
                else:
                    status.set(f"Training failed: {data}")
        if state["playing"]:
            step()
        draw()
        root.after(delay.get(), tick)
    new_maze()
    tick()
    root.mainloop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=["demo", "train", "evaluate"], default="demo")
    parser.add_argument("--episodes", type=int, default=30000)
    args = parser.parse_args()
    if not 1 <= args.episodes < 1_000_000:
        parser.error("episodes must be between 1 and 999999")
    if args.command == "train":
        _, report = train(args.episodes, progress=lambda row: print(row) if row["episode"] % 1000 == 0 else None)
        print(json.dumps(report, indent=2))
        print(f"Saved: {POLICY}")
    elif args.command == "evaluate":
        print(json.dumps(evaluate(load()), indent=2))
    else:
        demo()


if __name__ == "__main__":
    main()
