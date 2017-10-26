from rllab.envs.base import Env
from rllab.spaces import Box
from rllab.envs.base import Step
import numpy as np
from copy import deepcopy

class PointEnvRandGoalOracleNoise(Env):
    def __init__(self, goal=None,**kwargs):
        # TODO - call super class init?
        self._goal = goal
        if goal is None:
            self.set_at_init = False
        else:
            self.set_at_init = True
        if 'noise' in kwargs:
            self.action_noise = kwargs['noise']
        else:
            self.action_noise = 0
        self.debugcounter = 0

    @property
    def observation_space(self):
        return Box(low=-np.inf, high=np.inf, shape=(4,))

    @property
    def action_space(self):
        return Box(low=-0.1, high=0.1, shape=(2,))

    def sample_goals(self, num_goals):
        return np.random.uniform(-0.5, 0.5, size=(num_goals, 2, ))

    def reset(self, reset_args=None):
        if type(reset_args) is dict:
            goal = reset_args['goal']
            noise = reset_args['noise']
            if self.action_noise != noise:
                print("debug, action noise changing")
                self.action_noise = noise

        else:
            goal = reset_args
        if goal is not None:
            self._goal = goal
        elif not self.set_at_init:
            self._goal = np.random.uniform(-0.5, 0.5, size=(2,))

        self._state = self._goal + np.random.uniform(-0.5, 0.5, size=(2,)) #(0, 0)
        observation = np.copy(self._state)
        return np.r_[observation, np.copy(self._goal)]

    def step(self, action):
        action = action + np.random.normal(0., self.action_noise, size=action.shape)
    #    if self.debugcounter % 1000 == 0:
    #        print("debug1 noise", self.action_noise)
    #    self.debugcounter += 1
        self._state = self._state + action
        x, y = self._state
        x -= self._goal[0]
        y -= self._goal[1]
        reward = - (x ** 2 + y ** 2) ** 0.5
     #   done = abs(x) < 0.01 and abs(y) < 0.01
        done = False
        next_observation = np.r_[np.copy(self._state), np.copy(self._goal)]
        return Step(observation=next_observation, reward=reward, done=done)

    def render(self):
        print('current state:', self._state)

    def clip_goal_from_obs(self, paths):
        paths_copy = deepcopy(paths)
        for path in paths_copy:
            clipped_obs = path['observations'][:,:-2]  #[obs[:-2] for obs in path['observations']]
            path['observations'] = clipped_obs
        return paths_copy


