"""Propositional STRIPS-style breadth-first planner for a tiny warehouse."""
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class Action:
    name: str
    positive: frozenset
    negative: frozenset
    add: frozenset
    delete: frozenset

    def applicable(self, state):
        return self.positive <= state and self.negative.isdisjoint(state)

    def apply(self, state):
        return (state - self.delete) | self.add


def warehouse_actions(include_pickup=True):
    actions = [
        Action("Move(A,B)", frozenset({"At(Robot,A)"}), frozenset(), frozenset({"At(Robot,B)"}), frozenset({"At(Robot,A)"})),
        Action("Move(B,A)", frozenset({"At(Robot,B)"}), frozenset(), frozenset({"At(Robot,A)"}), frozenset({"At(Robot,B)"})),
        Action("Move(B,C)", frozenset({"At(Robot,B)"}), frozenset(), frozenset({"At(Robot,C)"}), frozenset({"At(Robot,B)"})),
        Action("Move(C,B)", frozenset({"At(Robot,C)"}), frozenset(), frozenset({"At(Robot,B)"}), frozenset({"At(Robot,C)"})),
        Action("Drop(Package,C)", frozenset({"At(Robot,C)", "Holding(Package)"}), frozenset(), frozenset({"At(Package,C)"}), frozenset({"Holding(Package)"})),
    ]
    if include_pickup:
        actions.insert(4, Action("PickUp(Package,A)", frozenset({"At(Robot,A)", "At(Package,A)"}), frozenset(), frozenset({"Holding(Package)"}), frozenset({"At(Package,A)"})))
    return actions


def bfs(initial, goal, actions):
    initial, goal = frozenset(initial), frozenset(goal)
    q, parent = deque([initial]), {initial: None}
    while q:
        state = q.popleft()
        if goal <= state:
            chain = []
            while parent[state] is not None:
                prev, action = parent[state]; chain.append((action, state)); state = prev
            return list(reversed(chain))
        for action in actions:
            if action.applicable(state):
                nxt = action.apply(state)
                if nxt not in parent: parent[nxt] = (state, action); q.append(nxt)
    return None


if __name__ == "__main__":
    init = {"At(Robot,A)", "At(Package,A)"}
    goal = {"At(Package,C)"}
    for label, actions in (("solvable", warehouse_actions()), ("no pickup", warehouse_actions(False))):
        result = bfs(init, goal, actions)
        print(label, "no plan" if result is None else [a.name for a, _ in result])
        if result:
            for i, (a, s) in enumerate(result, 1): print(i, a.name, sorted(s))
