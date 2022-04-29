from collections import defaultdict
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
    V: dict[tuple[int, int], float],
    env: GridWorld,
    gamma: float = 0.9,
) -> dict[tuple[int, int], float]:
    for state in env.states():
        if state == env.goal_state:
            # ゴールの価値関数は0
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


def value_iter_onestep(
    V: dict[tuple[int, int], float], env: GridWorld, gamma: float
) -> dict[tuple[int, int], float]:
    for state in env.states():
        if state == env.goal_state:
            # ゴールの価値関数は0
            V[state] = 0
            continue

        action_values = []
        for action in env.actions():
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            value = r + gamma * V[next_state]
            action_values.append(value)

        V[state] = max(action_values)
    return V


def policy_eval(
    pi: dict[tuple[int, int], dict],
    V: dict[tuple[int, int], float],
    env: GridWorld,
    gamma: float = 0.9,
    threshold: float = 0.001,
) -> dict[tuple[int, int], float]:

    while True:
        old_V = V.copy()
        V = eval_onestep(pi, V, env, gamma)

        # compute max(delta by updating V)
        delta = max(abs(V[state] - old_V[state]) for state in V.keys())

        if delta < threshold:
            break
    return V


def value_iter(
    V: dict[tuple[int, int], float],
    env: GridWorld,
    gamma: float,
    threshold: float = 0.001,
    is_render: bool = False,
) -> dict[tuple[int, int], float]:
    while True:
        if is_render:
            env.render_v(V)

        old_V = V.copy()
        V = value_iter_onestep(V, env, gamma)

        # compute max(delta by updating V)
        delta = max(abs(V[state] - old_V[state]) for state in V.keys())

        if delta < threshold:
            break
    return V


def argmax(d: dict):
    return max(d, key=d.get)


def greedy_policy(
    V: dict[tuple[int, int], float], env: GridWorld, gamma: float
) -> dict[tuple[int, int], dict]:
    pi = {}

    for state in env.states():

        action_values = {}

        for action in env.actions():
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            value = r + gamma * V[next_state]
            action_values[action] = value

            # greedy action
            max_action = argmax(action_values)
            action_probs = {i: 0.0 for i in range(4)}
            action_probs[max_action] = 1.0
            pi[state] = action_probs

    return pi


def policy_iter(
    env: GridWorld, gamma: float, threshold: float = 0.001, is_render: bool = False
) -> dict[tuple[int, int], dict]:
    pi = defaultdict(lambda: {i: 0.25 for i in range(4)})
    V: dict[tuple[int, int], float] = defaultdict(lambda: 0.0)

    while True:
        V = policy_eval(pi, V, env, gamma, threshold)
        new_pi = greedy_policy(V, env, gamma)

        if is_render:
            env.render_v(V, pi)

        if new_pi == pi:
            break
        pi = new_pi

    return pi
