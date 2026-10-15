import json
import unittest
from maze_car import Maze, generate, DIRS, ROOT, ACTIONS, evaluate, load


class BranchingMazeTests(unittest.TestCase):
    def test_connected_with_junctions_deadends_and_some_loops(self):
        cyclic = 0
        for seed in range(500):
            floor, start, goal, _ = generate(seed)
            degree = {p: sum((p[0]+dx, p[1]+dy) in floor for dx, dy in DIRS)
                      for p in floor}
            self.assertGreaterEqual(max(degree.values()), 3)
            self.assertGreaterEqual(sum(v == 1 for v in degree.values()), 2)
            seen, pending = {start}, [start]
            while pending:
                x, y = pending.pop()
                for dx, dy in DIRS:
                    p = (x+dx, y+dy)
                    if p in floor and p not in seen:
                        seen.add(p)
                        pending.append(p)
            self.assertEqual(seen, floor)
            self.assertNotEqual(start, goal)
            self.assertTrue(all(0 <= x < 9 and 0 <= y < 9 for x,y in floor))
            cyclic += sum(degree.values()) // 2 >= len(floor)
        self.assertGreater(cyclic, 0)
        self.assertLess(cyclic, 500)

    def test_observation_does_not_leak_hidden_map_or_goal(self):
        env = Maze(9)
        original = env.observation()
        hidden = env.floor - set(env.nearby()) - {env.pos, env.goal}
        replacement = next(iter(hidden))
        env.goal = replacement
        self.assertEqual(original, env.observation())
        env.floor.remove(replacement)
        self.assertEqual(original, env.observation())

    def test_visit_memory_and_backtracking(self):
        env = Maze(1)
        start = env.pos
        env.step(0)
        env.step(1)
        env.step(1)
        self.assertEqual(env.observation()[1], 2)
        _, reward, _ = env.step(0)
        self.assertEqual(env.pos, start)
        self.assertEqual(env.visit_counts[start], 2)
        self.assertEqual(env.backtracks, 1)
        self.assertEqual(reward, -0.04)

    def test_wall_collision(self):
        env = Maze(1)
        while env.nearby()[1] in env.floor:
            env.step(1)
        start = env.pos
        _, reward, _ = env.step(0)
        self.assertEqual(start, env.pos)
        self.assertEqual(reward, -1)

    def test_recordings_match_actual_transitions(self):
        payload = json.loads((ROOT / 'outputs/training_timeline.json').read_text())
        self.assertEqual(len(payload['traces']), 10)
        self.assertEqual(payload['traces'][0]['episode'], 1)
        self.assertEqual(payload['traces'][-1]['episode'], payload['episodes'])
        for trace in payload['traces']:
            env = Maze(trace['maze_seed'])
            for frame in trace['frames'][1:]:
                _, reward, _ = env.step(ACTIONS.index(frame['action']))
                self.assertEqual(list(env.pos), frame['pos'])
                self.assertEqual(env.heading, frame['heading'])
                self.assertEqual(env.collisions, frame['collisions'])
                self.assertEqual(reward, frame['reward'])
            self.assertTrue(env.done)
            self.assertEqual(env.success, trace['success'])

    def test_policy_on_separate_seeds_without_mutation(self):
        q = load()
        snapshot = json.dumps(q, sort_keys=True)
        report = evaluate(q, 300, seed=3_000_000)
        self.assertGreaterEqual(report['success_rate'], 0.9)
        self.assertEqual(snapshot, json.dumps(q, sort_keys=True))


if __name__ == '__main__':
    unittest.main()
