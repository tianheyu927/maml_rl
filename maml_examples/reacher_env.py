import numpy as np
from gym import utils
from rllab.core.serializable import Serializable
#from gym.envs.mujoco import mujoco_env  # this was originally here
from rllab.envs.mujoco.mujoco_env import MujocoEnv
from rllab.misc.overrides import overrides
from rllab.envs.base import Step

from rllab.envs.base import Env
from rllab.spaces import Box
from rllab.envs.env_spec import EnvSpec
from maml_examples.reacher_vars import ENV_OPTIONS

class ReacherEnv(MujocoEnv, Serializable):
    def __init__(self, option='g200nfj', *args, **kwargs):
        self.goal = None
        #utils.EzPickle.__init__(self)
        print("using env option ", ENV_OPTIONS[option])
        MujocoEnv.__init__(self, file_path=ENV_OPTIONS[option])
        Serializable.__init__(self, *args, **kwargs)

    def get_current_obs(self):
        theta = self.model.data.qpos.flat[:2]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.model.data.qvel.flat[:2],
            self.get_body_com("fingertip"),
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



    # def _step(self, a):
    #     vec = self.get_body_com("fingertip")-self.get_body_com("target")
    #     reward_dist = - np.linalg.norm(vec)
    #     reward_ctrl = - np.square(a).sum()
    #     reward = reward_dist + 0.0 * reward_ctrl
    #     self.do_simulation(a, self.frame_skip)
    #     ob = self._get_obs()
    #     done = False
    #     return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    # def reset_model(self):
    #     qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos # TODO, this seems to add a horizontal and vertical vector, resulting in a 4x4 matrix
    #     while True:
    #         self.goal = self.np_random.uniform(low=-.2, high=.2, size=2)
    #         if np.linalg.norm(self.goal) < 0.21:  # original gym env had norm limit of 2, which is 10x the reach of the arm
    #             break
    #     qpos[-2:] = self.goal
    #     qvel = self.init_qvel # no noise in velocities + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
    #     qvel[-2:] = 0
    #     self.set_state(qpos, qvel)


        return self._get_obs()


    # def _get_obs(self):
    #     theta = self.model.data.qpos.flat[:2]
    #     return np.concatenate([
    #         np.cos(theta),
    #         np.sin(theta),
    #         self.model.data.qpos.flat[2:],
    #         self.model.data.qvel.flat[:2],
    #         self.get_body_com("fingertip") - self.get_body_com("target")
    #     ])


    def sample_goals(self, num_goals):  # this actually samples whole initial qpos states
        goals_list = []
        for _ in range(num_goals):
            while True:
                newgoal = self.np_random.uniform(low=-.2, high=.2, size=(2, 1))
                if np.linalg.norm(newgoal) < 0.21:
                    break
            state_and_goal = np.concatenate([
                np.zeros(np.shape(self.init_qpos[:-2])),
                newgoal
            ])
            goals_list.append(state_and_goal)
        return np.array(goals_list)

    @property
    def spec(self):
        return EnvSpec(
            observation_space=self.observation_space,
            action_space=self.action_space,
        )

# def fingertip(obs):
#     a,b  = obs
#     elbow = np.array([np.cos(a),np.sin(a)]) * 0.1
#     fingertip = elbow + np.array([np.cos(a+b), np.sin(a+b)]) * 0.11
#     return fingertip