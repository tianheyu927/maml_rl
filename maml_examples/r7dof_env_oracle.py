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

class Reacher7DofMultitaskEnvOracle(Serializable):
    def __init__(self, xml_file=None, distance_metric_order=None, distractors=True, *args, **kwargs):
        self.goal = None
        if 'noise' in kwargs:
            noise = kwargs['noise']
        else:
            noise = 0.0
        self.include_distractors=distractors
        if self.include_distractors:
            self.shuffle_order = [[0,1,2],[1,2,0],[2,0,1]][2]
            # self.shuffle_order = rd.sample([[0,1,2],[1,2,0],[2,0,1]],1)[0]

        if xml_file is None:
            if not self.include_distractors:
                xml_file = '/home/rosen/maml_rl/vendor/mujoco_models/r7dof_versions/reacher_7dof.xml'
            else:
                xml_file = '/home/rosen/maml_rl/vendor/mujoco_models/r7dof_versions/reacher_7dof_2distr_%s%s%s.xml'%tuple(self.shuffle_order)

        print("xml file", xml_file)
        self.mujoco = mujoco_env.MujocoEnv(file_path=xml_file,action_noise=noise)
        self.action_space = self.mujoco.action_space
        self.get_viewer = self.mujoco.get_viewer
        self.log_diagnostics = self.mujoco.log_diagnostics
        Serializable.__init__(self, *args, **kwargs)
        # self.viewer_setup()
        # Serializable.quick_init(self, locals())
        # mujoco_env.MujocoEnv.__init__(
        #     self,
        #     file_path='r7dof_versions/reacher_7dof.xml',   # You probably need to change this
        #     action_noise=noise,
        #     #frame_skip = 5
        # )

    def viewer_setup(self):
        if self.mujoco.viewer is None:
            self.mujoco.start_viewer()
        self.mujoco.viewer.cam.trackbodyid = -1
        self.mujoco.viewer.cam.distance = 1.2
        self.mujoco.viewer.cam.azimuth = -90
        self.mujoco.viewer.cam.elevation = -60
        self.mujoco.viewer.cam.lookat = (0.0,0.0, 0.0)

    def get_current_obs(self):
        return self._get_obs()

    def get_current_image_obs(self):
        # image = self.mujoco.get_viewer().get_image()
        self.viewer_setup()
        self.mujoco.render()
        image = self.mujoco.viewer.get_image()
        pil_image = Image.frombytes('RGB', (image[1], image[2]), image[0])
        pil_image = pil_image.resize((64,64), Image.ANTIALIAS)
        pil_image = pil_image.crop((0,14,64,46))
        # pil_image.save("/home/rosen/temp12/pil_image.bmp")
        image = np.flipud(np.array(pil_image))
        # print("debug,norm of image", np.linalg.norm(np.array(pil_image)))
        return image, np.concatenate([  #this is the oracle environment so no need for distractors
            self.mujoco.model.data.qpos.flat[:7],
            self.mujoco.model.data.qvel.flat[:7],
            self.mujoco.get_body_com('tips_arm'),
            self.mujoco.get_body_com('goal'),
            ])

    def step(self, action):
        self.mujoco.frame_skip = 5
        distance = np.linalg.norm( self.mujoco.get_body_com("tips_arm") - self.mujoco.get_body_com("goal"))
        reward = - distance
        self.mujoco.do_simulation(action, n_frames=self.mujoco.frame_skip)
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
        qpos = np.copy(self.mujoco.init_qpos)
        qvel = np.copy(self.mujoco.init_qvel) + self.mujoco.np_random.uniform(
            low=-0.005, high=0.005, size=self.mujoco.model.nv
        )

        if type(reset_args) is dict:
            new_goal_pos = reset_args['goal']
            noise = reset_args['noise']
            if self.mujoco.action_noise != noise:
                print("debug action noise changing from %s to %s" % (self.mujoco.action_noise, noise))
                self.mujoco.action_noise = noise
        else:
            new_goal_pos = reset_args

        if new_goal_pos is not None:
            if np.equal(self.goal,new_goal_pos).all():
                self.goal = new_goal_pos
            else:
                print("debug env changing")
                self.goal = new_goal_pos
                # self.shuffle_order = rd.sample([[0, 1, 2], [1, 2, 0], [2, 0, 1]], 1)[0]
                # xml_file = '/home/rosen/maml_rl/vendor/mujoco_models/r7dof_versions/reacher_7dof_2distr_%s%s%s.xml' % tuple(
                #     self.shuffle_order)
                # self.mujoco.stop_viewer()
                # self.mujoco.terminate()
                # self.mujoco = mujoco_env.MujocoEnv(file_path=xml_file, action_noise=noise)
                # self.viewer_setup()
        else:  # change goal between resets
            print("debug, resetting goal de novo, shouldn't happen during demo collection")
            if not self.include_distractors:
                self.goal = np.random.uniform(low=[-0.4,-0.4,-0.3],high=[0.4,0.0,-0.3]).reshape(3,1)
            else:
                self.goal = np.random.uniform(low=[-0.4,-0.4,-0.3],high=[0.4,0.0,-0.3]).reshape(3,1)
        # reset distractors even if goal is same
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
            self.mujoco.model.data.qpos.flat[:7],
            self.mujoco.model.data.qvel.flat[:7],
            self.mujoco.get_body_com("tips_arm"),
            self.mujoco.get_body_com('goal'),
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
    @property
    @overrides
    def observation_space(self):
        shp = self.get_current_obs().shape
        ub = BIG * np.ones(shp)
        return spaces.Box(ub * -1, ub)

    def render(self):
        self.mujoco.render()