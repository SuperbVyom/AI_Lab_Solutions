"""Goal-based warehouse navigation agent using breadth-first search."""
from collections import deque

GRID = [
    "#####################", "#S....#............G#", "#.##....##########..#",
    "#....##.............#", "#.######.###.#.###..#", "#........#..........#",
    "#####################",
]
MOVES = ((-1, 0, "Up"), (1, 0, "Down"), (0, -1, "Left"), (0, 1, "Right"))


def find(grid, symbol):
    for r, row in enumerate(grid):
        for c, ch in enumerate(row):
            if ch == symbol: return r, c
    raise ValueError(f"Missing {symbol}")


def plan(grid=GRID):
    start, goal = find(grid, "S"), find(grid, "G")
    q, parent, expanded = deque([start]), {start: None}, 0
    while q:
        p = q.popleft()
        if p == goal: break
        expanded += 1
        for dr, dc, action in MOVES:
            n = p[0]+dr, p[1]+dc
            if 0 <= n[0] < len(grid) and 0 <= n[1] < len(grid[n[0]]) and grid[n[0]][n[1]] != "#" and n not in parent:
                parent[n] = (p, action); q.append(n)
    if goal not in parent: return None
    states, actions, p = [], [], goal
    while p is not None:
        states.append(p)
        prev = parent[p]
        if prev is None: break
        p, action = prev; actions.append(action)
    return list(reversed(states)), list(reversed(actions)), expanded


if __name__ == "__main__":
    result = plan()
    if result is None: print("No path found")
    else: print("States:", result[0]); print("Actions:", result[1]); print("Moves:", len(result[1])); print("Expanded:", result[2])
