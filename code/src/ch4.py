from tkinter import W
import numpy as np
from typing import Iterator
from itertools import product
from common.gridworld_render import Renderer


class GridWorld:
    def __init__(self):
        self.action_space = list(range(4))
        self.action_meaning = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
        self.action_move_map = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        self.reward_map = np.array([[0, 0, 0, 1.0], [0, None, 0, -1.0], [0, 0, 0, 0]])

        self.goal_state = (0, 3)
        self.wall_state = (1, 1)
        start_state = (2, 0)
        self.start_state = start_state
        self.agent_state = start_state

    @property
    def height(self) -> int:
        return len(self.reward_map)

    @property
    def width(self) -> int:
        return len(self.reward_map[0])

    @property
    def shape(self) -> tuple:
        return self.reward_map.shape

    def actions(self) -> list[int]:
        return self.action_space

    def states(self) -> Iterator[tuple[int, int]]:
        for h, w in product(range(self.height), range(self.width)):
            yield (h, w)

    def next_state(self, state: tuple[int, int], action: int) -> tuple[int, int]:
        # 壁を無視したときの移動先
        move = self.action_move_map[action]
        next_state = (state[0] + move[0], state[1] + move[1])

        # y, xの順であることに注意
        ny, nx = next_state
        is_nx_out_of_range = nx < 0 or nx >= self.width
        is_ny_out_of_range = ny < 0 or ny >= self.height

        # 壁の判定
        if is_nx_out_of_range or is_ny_out_of_range:
            return state
        elif next_state == self.wall_state:
            return state

        return next_state

    def reward(
        self, state: tuple[int, int], action: int, next_state: tuple[int, int]
    ) -> int:
        return self.reward_map[next_state]

    def render_v(self, v=None, policy=None, print_value=True):
        renderer = Renderer(self.reward_map, self.goal_state, self.wall_state)
        renderer.render_v(v, policy, print_value)


def eval_onestep(
    pi: dict[tuple[int, int], dict],
    V: dict[tuple[int, int], int],
    env: GridWorld,
    gamma=0.9,
) -> dict[tuple[int, int], int]:
    for state in env.states():
        if state == env.goal_state:
            # ゴールの価値観数は0
            V[state] = 0
            continue

        action_probs = pi[state]
        new_V = 0

        # formula 4.3
        for action, action_prob in action_probs.items():
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            new_V += action_prob * (r + gamma * V[next_state])
        V[state] = new_V
    return V


def policy_eval(
    pi: dict[tuple[int, int], dict],
    V: dict[tuple[int, int], int],
    env: GridWorld,
    gamma=0.9,
    threshold=0.001,
) -> dict[tuple[int, int], int]:

    while True:
        old_V = V.copy()
        V = eval_onestep(pi, V, env, gamma)

        # compute max(delta by updating V)
        delta = max(abs(V[state] - old_V[state]) for state in V.keys())

        if delta < threshold:
            break
    return V
