from rllab.envs.base import Env
from rllab.spaces import Box
from rllab.envs.base import Step
import numpy as np


class PointEnvRandGoalExpert(Env):
    def __init__(self, goal=None, option="box"):  # Can set goal to test adaptation.
        if option not in ["box", "smart_box", "disk"]:
            assert False, "need to select box, smart_box or disk"
        self._option = option
        self._goal = goal

    @property
    def observation_space(self):
        return Box(low=-np.inf, high=np.inf, shape=(2,))

    @property
    def action_space(self):
        return Box(low=-0.1, high=0.1, shape=(2,))

    def sample_goals(self, num_goals):
        return np.random.uniform(-0.5, 0.5, size=(num_goals, 2, ))
        #return np.array([[-0.5, 0.5]] * num_goals)

    def reset(self, reset_args=None):
        goal = reset_args
        if goal is not None:
            self._goal = goal
        elif self._goal is None:
            # Only set a new goal if this env hasn't had one defined before.
            self._goal = np.random.uniform(-0.5, 0.5, size=(2,))
            #goals = [np.array([-0.5,0]), np.array([0.5,0])]
            #goals = np.array([[-0.5,0], [0.5,0],[0.2,0.2],[-0.2,-0.2],[0.5,0.5],[0,0.5],[0,-0.5],[-0.5,-0.5],[0.5,-0.5],[-0.5,0.5]])
            #self._goal = goals[np.random.randint(10)]

        self._state = (0, 0)
        observation = np.copy(self._state)
        return observation

    def step(self, action):
        if self._option == "box":
            expert_action = self.getExpertAction(self._state)
            self._state = self._state + action
            x, y = self._state
            x -= self._goal[0]
            y -= self._goal[1]

        elif self._option == "disk":
            expert_action = self.getExpertActionSmartBox(self._state)
            # NOTE: we're comparing the expert_action with the raw action, so no need to scale expert_action into
            # the disk.
            disk_action = sq_to_disk(action)
            self._state = self._state + disk_action
            x, y = self._state
            x -= self._goal[0]
            y -= self._goal[1]

        elif self._option == "smart_box":
            expert_action = self.getExpertActionSmartBox(self._state)
            self._state = self._state + action
            x, y = self._state
            x -= self._goal[0]
            y -= self._goal[1]

        reward = - (x ** 2 + y ** 2) ** 0.5
        done = abs(x) < 0.01 and abs(y) < 0.01
        #done = False
        next_observation = np.copy(self._state)
        return Step(observation=next_observation, reward=reward, done=done, goal=self._goal,
                    expert_actions=expert_action)

    def render(self):
        print('current state:', self._state)

    def getExpertAction(self, state):
        s0, s1 = state
        goal0, goal1 = self._goal
        ((low0, low1), (high0, high1)) = self.action_space.bounds
        a0 = max(low0, min(high0, goal0-s0))
        a1 = max(low1, min(high1, goal1 - s1))  # TODO: we should use np.clip here
        return np.array((a0,a1))*10 # need to scale up 10x because of normalized environments

    def getExpertActionSmartBox(self, state):
        s0, s1 = state
        goal0, goal1 = self._goal
        a0, a1 = goal0-s0, goal1-s1 # setting action to point in a straight line from state to goal
        a0, a1 = np.dot((a0,a1), 0.1/max(abs(a0),abs(a1))) #making sure each of them is within the action space bounds assuming [-0.1,0.1] square
        return np.array((a0, a1))

    def getExpertActionDisk(self, state):
        return sq_to_disk(self.getExpertActionSmartBox(state))

def sq_to_disk(action):
    # Shrinks a point's coordinates toward 0, in a way that transforms the [-1,1]X[-1,1] square to a disk with radius 1 around (0,0)
    a0, a1 = action
    x0, x1 = np.dot((a0, a1), min(1.0/abs(a0),1.0/abs(a1))) # x is a centrally projected onto the [-1,1] square
    a0, a1 = np.dot((a0,a1),1/np.sqrt(x0**2+x1**2)) # divide the action by the length of x
    return np.array((a0, a1))