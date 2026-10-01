# Search and A* Laboratory Report

## Problem formulation and agent design

| Component | Specification |
|---|---|
| State | Grid coordinate `(row, column)` of the robot |
| Actions | Up, Down, Left, Right |
| Transition | Add the action's unit offset if the destination is in bounds and not `#` |
| Initial state | `S = (1, 1)` (zero-based) |
| Goal | `G = (7, 15)` |
| Cost | 1 per move |

The state needs only the current cell because the static grid determines all legal moves and costs. An action is invalid if it leaves the grid or enters a wall. The environment is deterministic: a legal move has a known result. A solution is a sequence of legal moves ending at the goal. `search_agent.py` stores coordinates in the frontier, best-known path costs and parent pointers, and reconstructs the path at the goal.

## Results

| Case / strategy | Found? | Path length | Expanded states |
|---|---:|---:|---:|
| Supplied map, A* Manhattan | yes | 40 | 63 |
| Supplied map, BFS | yes | 40 | 63 |
| Adjacent start and goal | yes | 1 | 1 |
| Inaccessible goal map | no | - | 9 |
| A* with `h=0` | yes | 40 | 63 |
| A* Euclidean | yes | 40 | 63 |
| A* with `2h` | yes | 40 | 66 |
| Two-route map, A* | yes | 4 | 4 |
| Two-route map, BFS | yes | 4 | 8 |

The original A* route, expressed as moves, is `Right×4, Down×4, Right×8, Up×2, Left×6, Up×2, Right×8, Down×6` (40 moves). BFS also found an optimal 40-move path. In this layout the admissible Manhattan heuristic did not reduce the measured expansion count; ties and corridor structure largely determine the order. `h=0` is uniform-cost search and reduces to BFS for unit costs. Euclidean distance is admissible here but also produced the same measured count. Doubling Manhattan is inadmissible; it still found an optimal path on this particular map, but that outcome is not guaranteed generally.

The no-solution example is `####### / #S....# / ###.### / #...#G# / #######`. The goal is enclosed by walls and the search terminates after exploring the reachable component. On the two-route map, A* follows the upper direct route and expands fewer states than BFS, while both find the shortest path.

## Code inspection

| Concept | Implementation |
|---|---|
| State / action / transition | coordinate tuples / `MOVES` / `neighbors` |
| Goal test | `cur == goal` |
| `g(n)` | `dist` / heap tuple's `g` |
| `h(n)` | local `h` function |
| `f(n)` | heap priority `g + h` |
| Frontier | `heapq` for A*, `deque` for BFS |
| Visited/best states | `dist`; stale heap entries are skipped |
| Path reconstruction | `parent` links in `_result` |

A* pops the lowest `f` priority. Manhattan distance is appropriate because a move changes one coordinate by one and diagonal moves are forbidden. It never overestimates the unobstructed number of required moves, so with unit costs it is admissible and A* returns an optimal path. An aggressive overestimate can bias the search toward a promising-looking route and lose the optimality guarantee.

## Prompt and validation

**Prompt used as the implementation specification:** “Implement A* for the supplied ASCII grid. Use `(row, column)` states, unit-cost cardinal moves, reject walls/out-of-bounds moves, calculate `f=g+h` with Manhattan distance, keep best costs and parent pointers, return the path, path length, and expanded count, and terminate cleanly when no route exists. Add BFS and parameterized zero, Euclidean, and scaled-Manhattan heuristic runs without changing the grid.”

The four required cases were run. A second map with two routes was also run under both algorithms; both returned a four-step shortest path. The path length and legality are independently checkable by the one-cell cardinal movement rules. The implementation improves on a common naive A* draft by allowing a state to be reopened when a lower path cost is found and ignoring stale heap entries.

## Reflection

Formulating the problem first makes the code's state, legal transitions, goal, and costs explicit; without that, even correct-looking search code may solve a different problem. A* is informed because it uses `h(n)` to estimate remaining distance when ordering states. Heuristic choice matters because it changes exploration order and whether optimality is guaranteed. An LLM can produce routine frontier and parent-map code quickly, but it can mishandle stale heap entries, reconstruct paths backward, or use a heuristic that overestimates. Testing on a one-step case, an unreachable goal, and multiple routes exposes these issues; accepting plausible output alone would not establish correctness. Implementing the agent clarifies that A* combines accumulated cost and estimated remaining cost, rather than simply choosing the cell closest to the goal.
