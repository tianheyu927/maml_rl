import numpy as np
from rllab.core.serializable import Serializable
from rllab.envs.mujoco.mujoco_env import MujocoEnv
from rllab.misc.overrides import overrides
from rllab.envs.base import Step

from rllab.envs.base import Env
from rllab.spaces import Box
from rllab.envs.env_spec import EnvSpec
from maml_examples.reacher_vars import ENV_OPTIONS, default_reacher_env_option


class ReacherEnv(MujocoEnv, Serializable):
    def __init__(self, option=default_reacher_env_option, *args, **kwargs):
        self.goal = None
        print("using env option ", ENV_OPTIONS[option])
        self.__class__.FILE = ENV_OPTIONS[option]
        super().__init__()
        Serializable.__init__(self, *args, **kwargs)

    def get_current_obs(self):
        theta = self.model.data.qpos.flat[:2]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.model.data.qvel.flat[:2],
            [np.cos(theta[0]) * 0.1 + np.cos(theta[0]+theta[1]) * 0.11],
            [np.sin(theta[0]) * 0.1 + np.sin(theta[0]+theta[1]) * 0.11],
        ])


    def step(self, action):
        self.frame_skip = 5
        vec = self.get_body_com("fingertip") - self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(action).sum()
        reward = reward_dist + reward_ctrl

        self.forward_dynamics(action)
        next_obs = self.get_current_obs()

        done = False
        return Step(next_obs, reward, done)

    def sample_goals(self, num_goals):
        goals_list = []
        for _ in range(num_goals):
            while True:
                #newgoal = np.array([-0.145, -0.145])
                newgoal = np.random.uniform(low=-.2, high=.2, size=2)
                if np.linalg.norm(newgoal) < 0.21:
                    break
            goals_list.append(newgoal)
        return np.array(goals_list)

    # this isn't used
    """
    def _step(self, a):
        vec = self.get_body_com("fingertip")-self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + 0.0 * reward_ctrl
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
    """

    @overrides
    def reset(self, reset_args=None, **kwargs):
        qpos = np.random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos.T[0]

        goal_pos = reset_args
        if goal_pos is not None:
            self.goal = goal_pos
        elif self.goal is None:  # keep the goal the same between resets
            while True:
                self.goal = np.random.uniform(low=-.2, high=.2, size=2)
                if np.linalg.norm(self.goal) < 0.21:
                    break

        qpos[-2:] = self.goal
        qvel = self.init_qvel.T[0] + np.random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        setattr(self.model.data, 'qpos', qpos)
        setattr(self.model.data, 'qvel', qvel)
        self.model.data.qvel = qvel
        self.model._compute_subtree()
        self.model.forward()
        self.current_com = self.model.data.com_subtree[0]
        self.dcom = np.zeros_like(self.current_com)
        return self.get_current_obs()



    # none of the following is called (due to differences between
    # rllab and gym)
    """
    def reset_model(self):
        qpos = np.random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos # TODO, this seems to add a horizontal and vertical vector, resulting in a 4x4 matrix
        while True:
            self.goal = np.random.uniform(low=-.2, high=.2, size=2)
            if np.linalg.norm(self.goal) < 0.21:  # original gym env had norm limit of 2, which is 10x the reach of the arm
                break
        qpos[-2:] = self.goal
        qvel = self.init_qvel # no noise in velocities + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)

        return self._get_obs()

    def _get_obs(self):
        theta = self.model.data.qpos.flat[:2]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.model.data.qpos.flat[2:],
            self.model.data.qvel.flat[:2],
            self.get_body_com("fingertip") - self.get_body_com("target")
        ])


    @property
    def spec(self):
        return EnvSpec(
            observation_space=self.observation_space,
            action_space=self.action_space,
        )
def fingertip(obs):
    a,b  = obs
    elbow = np.array([np.cos(a),np.sin(a)]) * 0.1
    fingertip = elbow + np.array([np.cos(a+b), np.sin(a+b)]) * 0.11
    return fingertip
"""