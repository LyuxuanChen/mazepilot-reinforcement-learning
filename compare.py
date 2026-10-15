"""Compare the saved policy with random actions on identical evaluation seeds."""
import json
from maze_car import evaluate, load, ROOT

if __name__ == '__main__':
    report = {
        'seed_start': 4_000_000,
        'random_actions': evaluate({}, episodes=500, seed=4_000_000),
        'learned_policy': evaluate(load(), episodes=500, seed=4_000_000),
    }
    print(json.dumps(report, indent=2))
    (ROOT / 'outputs/comparison.json').write_text(json.dumps(report, indent=2))
