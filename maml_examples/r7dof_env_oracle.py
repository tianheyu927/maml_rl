import numpy as np
import random as rd
from rllab.envs.mujoco import mujoco_env
from rllab.misc.overrides import overrides
from rllab.core.serializable import Serializable
from rllab.envs.base import Step
from PIL import Image

from copy import deepcopy

class Reacher7DofMultitaskEnvOracle(
    mujoco_env.MujocoEnv, Serializable
):
    def __init__(self, xml_file=None, distance_metric_order=None, distractors=False, *args, **kwargs):
        self.goal = None
        if 'noise' in kwargs:
            noise = kwargs['noise']
        else:
            noise = 0.0
        self.include_distractors=distractors
        if self.include_distractors:
            self.shuffle_order = rd.sample([[0,1,2],[1,2,0],[2,0,1]],1)[0]

        if xml_file is not None:
            self.__class__.FILE = xml_file
        else:
            if not self.include_distractors:
                self.__class__.FILE = 'r7dof_versions/reacher_7dof.xml'
            else:
                self.__class__.FILE = 'r7dof_versions/reacher_7dof_2distr_%s%s%s.xml'%tuple(self.shuffle_order)

        super().__init__(action_noise=noise)
        Serializable.__init__(self, *args, **kwargs)

        # Serializable.quick_init(self, locals())
        # mujoco_env.MujocoEnv.__init__(
        #     self,
        #     file_path='r7dof_versions/reacher_7dof.xml',   # You probably need to change this
        #     action_noise=noise,
        #     #frame_skip = 5
        # )

    def viewer_setup(self):
        if self.viewer is None:
            self.start_viewer()
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = -90
        self.viewer.cam.elevation = -60

    def get_current_obs(self):
        return self._get_obs()

    def get_current_image_obs(self):
        image = self.viewer.get_image()
        pil_image = Image.frombytes('RGB', (image[1], image[2]), image[0])
        pil_image = pil_image.resize((64,64), Image.ANTIALIAS)
        image = np.flipud(np.array(pil_image))
        return image, np.concatenate([  #this is the oracle environment so no need for distractors
            self.model.data.qpos.flat[:7],
            self.model.data.qvel.flat[:7],
            self.get_body_com('tips_arm'),
            self.get_body_com('goal'),
            ])

    def step(self, action):
        self.frame_skip = 5
        distance = np.linalg.norm(
            self.get_body_com("tips_arm") - self.get_body_com("goal")
        )
        reward = - distance
        self.forward_dynamics(action)
        # self.do_simulation(action, self.frame_skip)
        # next_obs = self.get_current_obs()
        next_img, next_obs = self.get_current_image_obs()
        done = False
        return Step(observation=next_obs, reward=reward, done=done, img=next_img) #, dict(distance=distance)

    def sample_goals(self, num_goals):
        goals_list = []
        for _ in range(num_goals):
            newgoal = np.random.uniform(low=[-0.4,-0.4,-0.3],high=[0.4,0.0,-0.3]).reshape(3,1)
            goals_list.append(newgoal)
        return np.array(goals_list)


    @overrides
    def reset(self, reset_args=None, **kwargs):
        qpos = np.copy(self.init_qpos)
        qvel = np.copy(self.init_qvel) + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )

        if type(reset_args) is dict:
            goal_pos = reset_args['goal']
            noise = reset_args['noise']
            if self.action_noise != noise:
                print("debug action noise changing")
                self.action_noise = noise
        else:
            goal_pos = reset_args

        if goal_pos is not None:
            self.goal = goal_pos
        else:  # change goal between resets
            if not self.include_distractors:
                self.goal = np.random.uniform(low=[-0.4,-0.4,-0.3],high=[0.4,0.0,-0.3]).reshape(3,1)
            else:
                self.goal = np.random.uniform(low=[-0.4,-0.4,-0.3],high=[0.4,0.0,-0.3]).reshape(3,1)
                self.distract1 = np.random.uniform(low=[-0.4,-0.4,-0.3],high=[0.4,0.0,-0.3]).reshape(3,1)
                self.distract2 = np.random.uniform(low=[-0.4,-0.4,-0.3],high=[0.4,0.0,-0.3]).reshape(3,1)
                qpos[-14:-11] = self.distract1
                qpos[-21:-18] = self.distract2
        qpos[-7:-4] = self.goal
        qvel[-7:] = 0
        setattr(self.model.data, 'qpos', qpos)
        setattr(self.model.data, 'qvel', qvel)
        self.model.data.qvel = qvel
        self.model._compute_subtree()
        self.model.forward()
        self.current_com = self.model.data.com_subtree[0]
        self.dcom = np.zeros_like(self.current_com)
        return self.get_current_obs()


    def clip_goal_from_obs(self, paths):
        paths_copy = deepcopy(paths)
        for path in paths_copy:
            clipped_obs = path['observations'][:, :-3]
            path['observations'] = clipped_obs
        return paths_copy


    # def reset_model(self):
    #     qpos = np.copy(self.init_qpos)
    #     qvel = np.copy(self.init_qvel) + self.np_random.uniform(
    #         low=-0.005, high=0.005, size=self.model.nv
    #     )
    #     print(np.shape(qpos[-7:-4]))
    #     qpos[-7:-4] = self._desired_xyz
    #     qvel[-7:] = 0
    #     self.set_state(qpos, qvel)
    #     return self._get_obs()

    def _get_obs(self):
        # if self.include_distractors:
        #     return np.concatenate([
        #         self.model.data.qpos.flat[:7],
        #         self.model.data.qvel.flat[:7],
        #         self.get_body_com("tips_arm"),
        #         self.get_body_com("distractor2"),
        #         self.get_body_com("distractor1"),
        #         self.get_body_com("goal"),
        #     ])
        # else:
        # this is the oracle environment, so no need for distractors
        return np.concatenate([
            self.model.data.qpos.flat[:7],
            self.model.data.qvel.flat[:7],
            self.get_body_com("tips_arm"),
            self.get_body_com('goal'),
        ])

    # def _step(self, a):
    #     distance = np.linalg.norm(
    #         self.get_body_com("tips_arm") - self.get_body_com("goal")
    #     )
    #     reward = - distance
    #     self.do_simulation(a, self.frame_skip)
    #     ob = self._get_obs()
    #     done = False
    #     return ob, reward, done, dict(distance=distance)

    # def log_diagnostics(self, paths):
    #     pass

