"""Grid search implementations for the warehouse search laboratory."""
from heapq import heappush, heappop
from collections import deque
from math import hypot

WAREHOUSE = [
    "#################", "#S....#.........#", "#.###.#.#######.#",
    "#...#.#.......#.#", "###.#.#######.#.#", "#...#.........#.#",
    "#.###########.#.#", "#.............#G#", "#################",
]
MOVES = ((-1, 0, "Up"), (1, 0, "Down"), (0, -1, "Left"), (0, 1, "Right"))


def locate(grid, marker):
    return next((r, c) for r, row in enumerate(grid) for c, ch in enumerate(row) if ch == marker)


def solve(grid, algorithm="astar", heuristic="manhattan", scale=1):
    start, goal = locate(grid, "S"), locate(grid, "G")
    def h(p):
        dr, dc = abs(p[0]-goal[0]), abs(p[1]-goal[1])
        if heuristic == "zero": return 0
        if heuristic == "euclidean": return hypot(dr, dc)
        return scale * (dr + dc)
    parent, dist, expanded = {}, {start: 0}, 0
    if algorithm == "bfs":
        frontier = deque([start])
        while frontier:
            cur = frontier.popleft()
            if cur == goal: return _result(start, goal, parent, dist[cur], expanded)
            expanded += 1
            for nxt, _ in neighbors(grid, cur):
                if nxt not in dist:
                    dist[nxt] = dist[cur] + 1; parent[nxt] = cur; frontier.append(nxt)
        return None, None, expanded
    frontier, serial = [], 0
    heappush(frontier, (h(start), serial, 0, start))
    while frontier:
        _, _, g, cur = heappop(frontier)
        if g != dist.get(cur): continue
        if cur == goal: return _result(start, goal, parent, g, expanded)
        expanded += 1
        for nxt, _ in neighbors(grid, cur):
            ng = g + 1
            if ng < dist.get(nxt, float("inf")):
                dist[nxt] = ng; parent[nxt] = cur; serial += 1
                heappush(frontier, (ng + h(nxt), serial, ng, nxt))
    return None, None, expanded


def neighbors(grid, p):
    for dr, dc, action in MOVES:
        q = p[0]+dr, p[1]+dc
        if 0 <= q[0] < len(grid) and 0 <= q[1] < len(grid[q[0]]) and grid[q[0]][q[1]] != "#":
            yield q, action


def _result(start, goal, parent, length, expanded):
    path = [goal]
    while path[-1] != start: path.append(parent[path[-1]])
    path.reverse()
    return path, length, expanded


def main():
    for name, grid, alg, heur, scale in [
        ("original A*", WAREHOUSE, "astar", "manhattan", 1),
        ("original BFS", WAREHOUSE, "bfs", "zero", 1),
        ("trivial", ["#####", "#SG##", "#####"], "astar", "manhattan", 1),
        ("unreachable", ["#######", "#S....#", "###.###", "#...#G#", "#######"], "astar", "manhattan", 1),
        ("h=0", WAREHOUSE, "astar", "zero", 1),
        ("euclidean", WAREHOUSE, "astar", "euclidean", 1),
        ("2h", WAREHOUSE, "astar", "manhattan", 2),
        ("two-route map A*", ["#######", "#S...G#", "#.....#", "#######"], "astar", "manhattan", 1),
        ("two-route map BFS", ["#######", "#S...G#", "#.....#", "#######"], "bfs", "zero", 1),
    ]:
        path, length, expanded = solve(grid, alg, heur, scale)
        print(name, "found", path is not None, "length", length, "expanded", expanded, "path", path)


if __name__ == "__main__": main()
