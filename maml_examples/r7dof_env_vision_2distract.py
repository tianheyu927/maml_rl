import numpy as np
import random as rd
from rllab.envs.mujoco import mujoco_env
from rllab.misc.overrides import overrides
from rllab.core.serializable import Serializable
from rllab.envs.base import Step
from PIL import Image
from rllab import spaces

from copy import deepcopy
BIG = 1e6


class Reacher7Dof2DistractVisionEnv(Serializable):
    def __init__(self, xml_file=None, goal_num=None, distance_metric_order=None, distractors=True, *args, **kwargs):
        self.goal_num = np.random.choice([0,1,2])
        self.shuffle_order = [[0,1,2],[1,2,0],[2,0,1]][self.goal_num]

        self.include_distractors=distractors
        assert distractors==True, "not supported"


        if xml_file is None:
                xml_file = '/home/rosen/maml_rl/vendor/mujoco_models/r7dof_versions/reacher_7dof_2distr_%s%s%s.xml'%tuple(self.shuffle_order)

        print("xml file", xml_file)
        self.mujoco = mujoco_env.MujocoEnv(file_path=xml_file)
        self.viewer_setup()
        self.action_space = self.mujoco.action_space
        self.get_viewer = self.mujoco.get_viewer
        self.log_diagnostics = self.mujoco.log_diagnostics
        Serializable.__init__(self, *args, **kwargs)
        # self.reset()  # resetting without a goal rearranges the objects without changing the shuffle order or xml file

    def viewer_setup(self):
        if self.mujoco.viewer is None:
            self.mujoco.start_viewer()
        self.mujoco.viewer.cam.trackbodyid = -1
        self.mujoco.viewer.cam.distance = 1.2
        self.mujoco.viewer.cam.azimuth = -90
        self.mujoco.viewer.cam.elevation = -60
        self.mujoco.viewer.cam.lookat = (0.0,0.0, 0.0)
    #
    # def get_current_obs(self):
    #     return np.concatenate([
    #         self.mujoco.model.data.qpos.flat[:7],
    #         self.mujoco.model.data.qvel.flat[:7],
    #         self.mujoco.get_body_com("tips_arm"),
    #     ])

    def get_current_image_obs(self):
        # self.viewer_setup()
        self.mujoco.render()
        image = self.mujoco.viewer.get_image()
        pil_image = Image.frombytes('RGB', (image[1], image[2]), image[0])
        pil_image = pil_image.resize((64,64), Image.ANTIALIAS)
        pil_image = pil_image.crop((0,14,64,46))
        image = np.flipud(np.array(pil_image))
        image = image.astype(np.float32)
        state = np.concatenate([
            self.mujoco.model.data.qpos.flat[:7],
            self.mujoco.model.data.qvel.flat[:7],
            self.mujoco.get_body_com('tips_arm'),
            ])
        return state, np.concatenate([image.flatten(),state])

    def step(self, action):
        self.mujoco.frame_skip = 5
        distance = np.linalg.norm( self.mujoco.get_body_com("tips_arm") - self.mujoco.get_body_com("goal"))
        reward = - distance
        # reward = 1.0 if distance < 0.1 else 0.0
        self.mujoco.forward_dynamics(action)
        # self.do_simulation(action, self.frame_skip)  <- forward dynamics produces better RL results as per AG 5/14/18
        next_state, next_obs = self.get_current_image_obs()
        # next_obs = self.get_current_obs()
        done = False
        # return Step(observation=next_obs, reward=reward, done=done, img=next_img)
        return Step(observation=next_obs, reward=reward, done=done, state=next_state)

    def sample_goals(self, num_goals):
        goals_list = []
        for _ in range(num_goals):
            newgoal = np.random.choice([0,1,2])
            goals_list.append(newgoal)
        return np.array(goals_list)


    @overrides
    def reset(self, reset_args=None, **kwargs):
        # print("debug,asked to reset with reset_args", reset_args)
        qpos = np.copy(self.mujoco.init_qpos)
        qvel = np.copy(self.mujoco.init_qvel) + 0.0*self.mujoco.np_random.uniform(low=-0.005, high=0.005, size=self.mujoco.model.nv)
        goal_num = reset_args
        if goal_num is not None:
            if self.goal_num != goal_num:
                self.goal_num = goal_num
                self.shuffle_order = [[0, 1, 2], [1, 2, 0], [2, 0, 1]][self.goal_num]
                # if self.mujoco.viewer is not None:
                #     self.mujoco.stop_viewer()
                    # self.mujoco.release()
                    # self.mujoco.terminate()
                # self.mujoco.terminate()
                xml_file = '/home/rosen/maml_rl/vendor/mujoco_models/r7dof_versions/reacher_7dof_2distr_%s%s%s.xml' % tuple(
                    self.shuffle_order)
                self.mujoco = mujoco_env.MujocoEnv(file_path=xml_file)
                self.viewer_setup()
        elif self.goal_num is None: # do not change color of goal or XML file between resets.
            self.goal_num = np.random.choice([0,1,2])
            self.shuffle_order = [[0,1,2],[1,2,0],[2,0,1]][self.goal_num]
            if self.mujoco.viewer is not None:
                self.mujoco.stop_viewer()
                self.mujoco.release()
                self.mujoco.terminate()
            xml_file = '/home/rosen/maml_rl/vendor/mujoco_models/r7dof_versions/reacher_7dof_2distr_%s%s%s.xml'%tuple(self.shuffle_order)
            self.mujoco = mujoco_env.MujocoEnv(file_path=xml_file)
            self.viewer_setup()
        self.goal = np.random.uniform(low=[-0.4, -0.4, -0.3], high=[0.4, 0.0, -0.3]).reshape(3, 1)
        self.distract1 = np.random.uniform(low=[-0.4,-0.4,-0.3],high=[0.4,0.0,-0.3]).reshape(3,1)
        self.distract2 = np.random.uniform(low=[-0.4,-0.4,-0.3],high=[0.4,0.0,-0.3]).reshape(3,1)
        qpos[-14:-11] = self.distract1
        qpos[-21:-18] = self.distract2
        qpos[-7:-4] = self.goal
        qvel[-7:] = 0
        setattr(self.mujoco.model.data, 'qpos', qpos)
        setattr(self.mujoco.model.data, 'qvel', qvel)
        self.mujoco.model.data.qvel = qvel
        self.mujoco.model._compute_subtree()
        self.mujoco.model.forward()
        self.current_com = self.mujoco.model.data.com_subtree[0]
        self.dcom = np.zeros_like(self.current_com)
        return self.get_current_image_obs()[1]
        # return self.get_current_image_obs()[1]



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

    # def _get_obs(self):
    #     return np.concatenate([
    #         self.model.data.qpos.flat[:7],
    #         self.model.data.qvel.flat[:7],
    #         self.get_body_com("tips_arm"),
    #     ])

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
    @property
    @overrides
    def observation_space(self):
        shp = self.get_current_image_obs()[1].shape
        # shp = self.get_current_image_obs()[1].shape
        ub = BIG * np.ones(shp)
        return spaces.Box(ub * -1, ub)

    def render(self):
        self.mujoco.render()