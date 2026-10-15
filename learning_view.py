"""Replay ten actual training trials, sampled across a Q-learning run."""
import json
import queue
import threading
import tkinter as tk
from tkinter import ttk
from maze_car import ROOT, DIRS, generate, train
from ui_style import apply_style


def main():
    root = tk.Tk()
    root.title("Branching Maze Car — ten moments of learning")
    apply_style(root)
    root.resizable(False, False)
    messages = queue.Queue()
    data = {"traces": [], "history": [], "frame": 0, "playing": True,
            "training": False, "tick": 0}
    ttk.Label(root, text="LEARNING TO EXPLORE BRANCHING MAZES",
              font=("Helvetica", 19, "bold")).pack(pady=(14, 4))
    ttk.Label(root, text="10 actual training episodes · read left to right, top to bottom · one shared policy learns between panels").pack()
    ttk.Label(root, text="Full mazes are for you. The car observes adjacent left/front/right cells and its visited-cell memory.").pack(pady=4)
    controls = ttk.Frame(root)
    controls.pack(pady=5)
    canvas = tk.Canvas(root, width=1060, height=650, bg="#101827", highlightthickness=0)
    canvas.pack(padx=10)
    status = tk.StringVar(value="Loading…")
    ttk.Label(root, textvariable=status, wraplength=1040).pack(pady=8)
    speed = tk.IntVar(value=35)
    tk.Scale(controls, from_=10, to=200, variable=speed, length=230,
             orient="horizontal", label="Milliseconds per action (lower = faster)").pack(side="right", padx=16)

    def draw():
        canvas.delete("all")
        for i, trace in enumerate(data["traces"]):
            ox, oy = 10 + (i % 5) * 210, 8 + (i // 5) * 270
            frames = trace["frames"]
            index = min(data["frame"], len(frames) - 1)
            frame = frames[index]
            floor, start, goal, _ = generate(trace["maze_seed"])
            visited = {tuple(f["pos"]) for f in frames[:index+1]}
            x, y = frame["pos"]
            visible = {(x, y)}
            for turn in (-1, 0, 1):
                dx, dy = DIRS[(frame["heading"] + turn) % 4]
                visible.add((x+dx, y+dy))
            canvas.create_text(ox, oy, anchor="nw", fill="white", font=("Helvetica", 12, "bold"),
                               text=f"{i+1:02d} · Episode {trace['episode']:,}")
            canvas.create_text(ox, oy+21, anchor="nw", fill="#acbbd1",
                               text=f"Seed {trace['maze_seed']} · explore {trace['epsilon']:.0%}")
            for gy in range(9):
                for gx in range(9):
                    p = (gx, gy)
                    color = "#28364b" if p not in floor else "#dfe7ef"
                    if p in visited:
                        color = "#89bfb8"
                    if p == goal:
                        color = "#3bdb8a"
                    canvas.create_rectangle(ox+gx*20, oy+44+gy*20,
                        ox+gx*20+19, oy+44+gy*20+19, fill=color,
                        outline="#f9c95c" if p in visible else color)
            cx, cy = ox+x*20+10, oy+44+y*20+10
            dx, dy = DIRS[frame["heading"]]
            canvas.create_polygon(cx+dx*8, cy+dy*8, cx-dx*6-dy*5,
                cy-dy*6+dx*5, cx-dx*6+dy*5, cy-dy*6-dx*5, fill="#fa694c")
            ended = index == len(frames)-1
            label = ("GOAL" if trace["success"] else "TIMEOUT") if ended else frame["action"]
            canvas.create_text(ox, oy+228, anchor="nw", fill="#60dfa3" if ended and trace["success"] else "#f9c95c",
                text=f"{label} · {index} actions · hits {frame['collisions']}")
            canvas.create_text(ox, oy+247, anchor="nw", fill="#acbbd1", font=("Helvetica", 10),
                text=f"r {frame['reward']:+.2f} · Q {frame['q_before']:.2f} → {frame['q_after']:.2f}")
        canvas.create_text(12, 555, anchor="nw", fill="white", text="Training success rate (100-episode blocks; includes exploratory mistakes)")
        history = data["history"]
        if len(history) > 1:
            points = []
            for i, row in enumerate(history):
                points.extend([45+i*990/(len(history)-1), 630-row["training_success_rate"]*48])
            canvas.create_line(*points, fill="#60dfa3", width=2)
            canvas.create_text(7, 582, anchor="w", fill="#acbbd1", text="100%")
            canvas.create_text(17, 630, anchor="w", fill="#acbbd1", text="0%")

    def accept(payload):
        data.update(traces=payload["traces"], history=payload["history"], frame=0, playing=True)
        status.set("Replaying recorded training trials side by side. Later trials share earlier learning; they are not ten independent agents.")

    def retrain():
        if data["training"]:
            return
        data.update(training=True, playing=False)
        status.set("Training a fresh policy and recording ten evenly spaced trials…")
        def worker():
            try:
                train(progress=lambda row: messages.put(("progress", row)))
                payload = json.loads((ROOT / "outputs/training_timeline.json").read_text())
                messages.put(("done", payload))
            except Exception as exc:
                messages.put(("error", str(exc)))
        threading.Thread(target=worker, daemon=True).start()

    def replay():
        if not data["training"]:
            data.update(frame=0, playing=True)

    def toggle():
        if not data["training"]:
            data["playing"] = not data["playing"]

    for label, command in [("Replay all 10", replay), ("Pause / resume", toggle), ("Train fresh + record", retrain)]:
        ttk.Button(controls, text=label, command=command).pack(side="left", padx=4)

    def tick():
        while not messages.empty():
            kind, payload = messages.get_nowait()
            if kind == "progress":
                status.set(f"Learning: episode {payload['episode']}/30000 · recent success {payload['training_success_rate']:.0%}")
            else:
                data["training"] = False
                if kind == "done":
                    accept(payload)
                else:
                    status.set(f"Training error: {payload}")
        draw()
        if data["playing"] and data["traces"]:
            maximum = max(len(t["frames"]) for t in data["traces"])-1
            if data["frame"] < maximum:
                data["frame"] += 1
            else:
                data["playing"] = False
                status.set("Replay complete. Compare collisions and actions across panels. A single later trial can still fail because training includes exploration. Click Replay all 10 to watch again.")
        root.after(speed.get(), tick)

    path = ROOT / "outputs/training_timeline.json"
    if path.exists():
        accept(json.loads(path.read_text()))
    else:
        retrain()
    tick()
    root.mainloop()


if __name__ == "__main__":
    main()
