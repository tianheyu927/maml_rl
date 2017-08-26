import numpy as np

from rllab.core.serializable import Serializable
from rllab.envs.base import Step
from rllab.envs.mujoco.mujoco_env import MujocoEnv
from rllab.misc import logger
from rllab.misc.overrides import overrides


def smooth_abs(x, param):
    return np.sqrt(np.square(x) + np.square(param)) - param


class PusherEnv(MujocoEnv, Serializable):

    FILE = None #'pusher.xml' #'/home/rosen/rllab_copy/vendor/local_mujoco_models/ensure_woodtable_distractor_pusher1.xml'

    def __init__(self, xml_file, *args, **kwargs):
        self.frame_skip = 5
        self.__class__.FILE = xml_file
        if 'distractors' in kwargs:
            self.include_distractors = kwargs['distractors']
        else:
            self.include_distractors = False
        #kwargs.pop('xml_file')
        super(PusherEnv, self).__init__(*args, **kwargs)
        self.frame_skip = 5
        Serializable.__init__(self, *args, **kwargs)

    def get_current_obs(self):
        if self.include_distractors:
            return np.concatenate([
                self.model.data.qpos.flat[:7],
                self.model.data.qvel.flat[:7],
                self.get_body_com("tips_arm"),
                self.get_body_com("distractor"),
                self.get_body_com("object"),
                self.get_body_com("goal"),
            ])
        else:
           return np.concatenate([
                self.model.data.qpos.flat[:7],
                self.model.data.qvel.flat[:7],
                self.get_body_com("tips_arm"),
                self.get_body_com("object"),
                self.get_body_com("goal"),
            ])


    #def get_body_xmat(self, body_name):
    #    idx = self.model.body_names.index(body_name)
    #    return self.model.data.xmat[idx].reshape((3, 3))

    def get_body_com(self, body_name):
        idx = self.model.body_names.index(body_name)
        return self.model.data.com_subtree[idx]

    def step(self, action):
        self.frame_skip = 5
        vec_1 = self.get_body_com("object") - self.get_body_com("tips_arm")
        vec_2 = self.get_body_com("object") - self.get_body_com("goal")
        reward_near = - np.linalg.norm(vec_1)
        reward_dist = - np.linalg.norm(vec_2)
        reward_ctrl = - np.square(action).sum()
        reward = reward_dist + 0.1 * reward_ctrl + 0.5 * reward_near

        self.forward_dynamics(action) # TODO - frame skip
        next_obs = self.get_current_obs()

        done = False
        return Step(next_obs, reward, done)

    @overrides
    def reset(self, init_state=None):
        self.frame_skip = 5
        qpos = self.init_qpos.copy()
        self.goal_pos = np.asarray([0, 0])

        while True:
            self.obj_pos = np.concatenate([
                    np.random.uniform(low=-0.3, high=0, size=1),
                    np.random.uniform(low=-0.2, high=0.2, size=1)])
            if np.linalg.norm(self.obj_pos - self.goal_pos) > 0.17:
                break

        if self.include_distractors:
            if self.obj_pos[1] < 0:
                y_range = [0.0, 0.2]
            else:
                y_range = [-0.2, 0.0]

            while True:
                self.distractor_pos = np.concatenate([
                    np.random.uniform(low=-0.3, high=0, size=1),
                    np.random.uniform(low=y_range[0], high=y_range[1], size=1)])
                if np.linalg.norm(self.distractor_pos - self.goal_pos) > 0.17 and np.linalg.norm(self.obj_pos - self.distractor_pos) > 0.1:
                    break
            qpos[-6:-4,0] = self.distractor_pos

        qpos[-4:-2,0] = self.obj_pos
        qpos[-2:,0] = self.goal_pos
        qvel = self.init_qvel + np.random.uniform(low=-0.005,
                high=0.005, size=self.model.nv)
        setattr(self.model.data, 'qpos', qpos)
        setattr(self.model.data, 'qvel', qvel)
        self.model.data.qvel = qvel
        self.model._compute_subtree()
        self.model.forward()

        self.current_com = self.model.data.com_subtree[0]
        self.dcom = np.zeros_like(self.current_com)
        return self.get_current_obs()

    @overrides
    def log_diagnostics(self, paths):
        pass
