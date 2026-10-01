# Goal-Based Agent: Warehouse Navigation

## Problem and design

The environment is the finite, fully observable grid printed in the lab. A state is the vehicle's row and column. Its goal is to reach the cell marked `G`; actions move one cell up, down, left, or right. The agent stores the current location and a search frontier with parent links, then reconstructs the route once the goal is found. The program in `goal_based_agent.py` uses breadth-first search (BFS), which is complete and returns a shortest route when each move costs one. It found the 20-move route and expanded 58 states before reaching the goal. The generated program worked on its first run on the supplied map; using A* instead could reduce exploration on some larger maps, though the search strategy remains appropriate here.

This is goal-based because the choice of action depends on a destination condition and a planned route through the map. A simple reflex agent would react to the current percept without maintaining a goal-directed plan. For a grid twice as large, BFS remains valid but may use much more memory and expand many more cells; A* with a useful admissible heuristic is often more efficient.

## Block diagram

```text
Grid map + current position + goal
                 |
                 v
      State and valid-neighbor generator
                 |
                 v
        BFS frontier / parent map
                 |
                 v
       Goal test -> reconstruct route
                 |
                 v
           Action sequence
```

## Run result

On the supplied map, BFS found a 20-move route:

`Right, Right, Right, Down, Right, Right, Right, Up, Right, Right, Right, Right, Right, Right, Right, Right, Right, Right, Right, Right`.

The row/column states are printed by the source file. Every step remains in bounds and outside `#`, and the last state is the goal at row 1, column 19 (zero-based indexing).

## Prompt and engineering reflection

**Specification prompt:** “Implement a goal-based agent for this exact ASCII warehouse. Treat a state as `(row, column)`, allow only up/down/left/right moves into non-wall cells, use BFS, detect `G`, reconstruct and print the actions and states, and report no path if unreachable. Do not change the map or movement rules.”

The implementation uses BFS because all moves have equal cost and the map is small. It ran successfully in this environment on the supplied map. The source was checked against the state/action rules and the path endpoints. No generated-code failure was observed, so no corrective prompt was needed; verification still matters because a route can appear plausible while crossing a wall or ending at the wrong location.
