import numpy as np
from gym import utils
from rllab.envs.mujoco import mujoco_env
from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides
from rllab import spaces

from rllab.envs.base import Step
import joblib
import os.path
# import mujoco_py
# from mujoco_py.mjlib import mjlib
from PIL import Image

BIG = 1e6


class PusherEnv(mujoco_env.MujocoEnv, Serializable):
    def __init__(self, xml_file=None, distractors=True, onehot=True, *args, **kwargs):
        utils.EzPickle.__init__(self)
        print("using xml_file", xml_file)
        if xml_file is None:
            xml_file = 'pusher.xml'

        self.__class__.FILE = xml_file
        self.include_distractors = distractors
        self.test_dir = "/home/kevin/rllab/pushing/test2_paired_push_demos_noimg/"
        self.train_dir = "/home/kevin/FaReLI_data/pushing/paired_push_demos_noimg/"
        self.xml_dir = "/home/kevin/gym/gym/envs/mujoco/assets/sim_push_xmls/"
        self.goal_num = 1
        self.test = False
        self.onehot=onehot
        self.onehot_dim = 5
        self.onehot_position = 0
        super(PusherEnv, self).__init__(*args, **kwargs)
        # import pdb; pdb.set_trace()
        # self = mujoco_env.MujocoEnv(file_path=xml_file)
        # self.viewer_setup()
        # self.reset()
        # self.observation_space = self.observation_space
        # self.action_space = self.action_space
        # self.goal_num, self.test = self.sample_goals(num_goals=1, test=False)[0]
        Serializable.__init__(self, *args, **kwargs)
        # self.__init__(self, file_path=xml_file)

    def viewer_setup(self):
        if self.viewer is None:
            self.start_viewer()
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 4.0

    def get_current_obs(self):
        return self._get_obs()

    def step(self, action):
        self.frame_skip = 5
        ob, reward, done, reward_dict = self._step(a=action)
        return Step(ob, reward, done)

    def sample_goals(self, num_goals, test=False):
        out = []
        for _ in range(num_goals):
            if not test:
                while True:
                    i = int(np.random.choice(1000,1))
                    if os.path.isfile(self.train_dir+str(i)+".pkl"):
                        out.append((i,False))
                        break
            else:
                while True:
                    i = int(np.random.choice(1000,1))
                    if os.path.isfile(self.test_dir+str(i)+".pkl"):
                        out.append((i,True))
                        break
        return np.array(out)

    @overrides
    def reset(self, reset_args=None, **kwargs):
        goal = reset_args
        if goal is not None:
            assert len(goal)==2, "wrong size goal"
            goal_num, test = goal
            if (goal_num != self.goal_num) or (test != self.test):
                if self.viewer is not None:
                    self.stop_viewer()
                self.terminate()
                self.goal_num, self.test = goal
                demo_path = (self.train_dir + str(self.goal_num) + ".pkl") if not self.test else (
                self.test_dir + str(self.goal_num) + ".pkl")
                demo_data = joblib.load(demo_path)
                xml_file = demo_data["xml"]
                xml_file = xml_file.replace("/root/code/rllab/vendor/mujoco_models/", self.xml_dir)
                print("debug,xml_file", xml_file)
                if int(xml_file[-5])%2==0:
                    print("retaining order")
                    self.shuffle_order=[0,1]
                else:
                    print("flipping order")
                    self.shuffle_order=[1,0]
                self = mujoco_env.MujocoEnv(file_path=xml_file)
        elif self.goal_num is None:  #if we already have a goal_num, we don't sample a new one, just reset the model
            self.goal_num, self.test = self.sample_goals(num_goals=1,test=False)[0]
            demo_path = (self.train_dir+str(self.goal_num)+".pkl") if not self.test else (self.test_dir+str(self.goal_num)+".pkl")
            demo_data = joblib.load(demo_path)
            xml_file = demo_data["xml"]
            xml_file = xml_file.replace("/root/code/rllab/vendor/mujoco_models/",self.xml_dir)

            if int(xml_file[-5]) % 2 == 0:
                print("retaining order")
                self.shuffle_order = [0, 1]
            else:
                print("flipping order")
                self.shuffle_order = [1, 0]
            self =  mujoco_env.MujocoEnv(file_path=xml_file)
            self.viewer_setup()
        self.reset_model()
        return self.get_current_obs()

    def _step(self, a):
        # normalize actions
        if self.action_space is not None:
            lb, ub = self.action_space.low, self.action_space.high
            a = lb + (a + 1.) * 0.5 * (ub - lb)
            a = np.clip(a, lb, ub)

        vec_1 = self.get_body_com("object") - self.get_body_com("tips_arm")
        vec_2 = self.get_body_com("object") - self.get_body_com("goal")

        reward_near = - np.linalg.norm(vec_1)
        reward_dist = - np.linalg.norm(vec_2)
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + 0.1 * reward_ctrl + 0.5 * reward_near

        self.do_simulation(a, n_frames=self.frame_skip)
        # extra added to copy rllab forward_dynamics.
        self.model.forward()

        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist,
                reward_ctrl=reward_ctrl)

    def reset_model(self):
        qpos = np.copy(self.init_qpos)

        self.goal_pos = np.asarray([0., 0.])
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
                        np.random.uniform(low=y_range[0], high=y_range[1], size=1)]).reshape(2,1)
                if np.linalg.norm(self.distractor_pos - self.goal_pos) > 0.17 and np.linalg.norm(self.obj_pos - self.distractor_pos) > 0.1:
                    break
            qpos[-6:-4] = self.distractor_pos
        qpos[-4:-2] = self.obj_pos.reshape(2,1)
        qpos[-2:] = self.goal_pos.reshape(2,1)
        qvel = self.init_qvel + np.random.uniform(low=-0.005,
                high=0.005, size=(self.model.nv))
        self.onehot_position = np.random.choice(range(self.onehot_dim))

        #qvel[-4:] = 0
        #self.set_state(qpos, qvel)
        #return self._get_obs()

        setattr(self.model.data, 'qpos', qpos)
        setattr(self.model.data, 'qvel', qvel)
        self.model.data.qvel = qvel
        self.model._compute_subtree()
        self.model.forward()
        self.current_com = self.model.data.com_subtree[0]
        self.dcom = np.zeros_like(self.current_com)
        return self._get_obs()

    def get_current_image_obs(self):
        image = self.viewer.get_image()
        pil_image = Image.frombytes('RGB', (image[1], image[2]), image[0])
        pil_image = pil_image.resize((125,125), Image.ANTIALIAS)
        image = np.flipud(np.array(pil_image))
        return image, np.concatenate([
            self.model.data.qpos.flat[:7],
            self.model.data.qvel.flat[:7],
            self.get_body_com('tips_arm'),
            self.get_body_com('goal'),
            ])


    def _get_obs(self):
        if self.include_distractors:
            # if self.shuffle_order[0] == 0:
            #     return np.concatenate([
            #         self.model.data.qpos.flat[:7],
            #         self.model.data.qvel.flat[:7],
            #         self.get_body_com("tips_arm"),
            #         self.get_body_com("distractor"),
            #         self.get_body_com("object"),
            #         self.get_body_com("goal"),
            #     ])
            # else:
            if self.onehot:
                extra = np.zeros(self.onehot_dim)
                if self.onehot_position == -1:
                    pass # we keep the vector zeroed out
                elif self.onehot_position in range(self.onehot_dim):
                    extra[self.onehot_position]=1.0
                else:
                    assert False, "invalid value of self.onehot_position"
                return np.concatenate([
                    self.model.data.qpos.flat[:7],
                    self.model.data.qvel.flat[:7],
                    self.get_body_com("tips_arm"),
                    self.get_body_com("object"),
                    self.get_body_com("distractor"),
                    self.get_body_com("goal"),
                    extra
                ])
            return np.concatenate([
                self.model.data.qpos.flat[:7],
                self.model.data.qvel.flat[:7],
                self.get_body_com("tips_arm"),
                self.get_body_com("object"),
                self.get_body_com("distractor"),
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

    def render(self):
        self.render()

    @property
    @overrides
    def observation_space(self):
        shp = self.get_current_obs().shape
        ub = BIG * np.ones(shp)
        return spaces.Box(ub * -1, ub)

    def get_viewer(self):
        return self.get_viewer()