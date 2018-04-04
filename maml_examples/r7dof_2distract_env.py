import numpy as np
from rllab.envs.mujoco import mujoco_env
from rllab.misc.overrides import overrides
from rllab.envs.base import Step



from rllab.core.serializable import Serializable


class Reacher7Dof2DistractEnv(
    mujoco_env.MujocoEnv, Serializable
):
    def __init__(self, distance_metric_order=None, *args, **kwargs):
        self.goal = None
        self.shuffled_order = None
        self.objects = ["goal","distract1","distract2"]
        self.__class__.FILE = 'r7dof_versions/reacher_7dof_2distract.xml'
        super().__init__()
        Serializable.__init__(self, *args, **kwargs)

        # Serializable.quick_init(self, locals())
        # mujoco_env.MujocoEnv.__init__(
        #     self,
        #     file_path='r7dof_versions/reacher_7dof.xml',   # You probably need to change this
        #     #frame_skip = 5
        # )
     #   self._desired_xyz = np.zeros((3,1))

    def viewer_setup(self):
        if self.viewer is None:
            self.start_viewer()
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 3.5
        self.viewer.cam.azimuth = -30

    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[:7],
            self.model.data.qvel.flat[:7],
            self.get_body_com("tips_arm"),
            self.get_body_com(self.objects[self.shuffled_order[0]]),
            self.get_body_com(self.objects[self.shuffled_order[1]]),
            self.get_body_com(self.objects[self.shuffled_order[2]]),
        ])

    def step(self, action):
        self.frame_skip = 5
        distance = np.linalg.norm(
            self.get_body_com("tips_arm") - self.get_body_com("goal")
        )
        reward = - distance
        # reward = 1.0 if distance < 0.1 else 0.0
        self.forward_dynamics(action)
        # self.do_simulation(action, self.frame_skip)
        next_obs = self.get_current_obs()
        done = False
        return Step(next_obs, reward, done) #, dict(distance=distance)

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

        goal = reset_args
        if goal is not None:
            self.goal = goal[0]
            self.distract1 = goal[1]
            self.distract2 = goal[2]
            self.shuffled_order = goal[3] # get a new shuffled order on a new task
        elif self.goal is None: # do not change goal between resets, only at initialization and when explicitly given a new goal
        # elif reset_args is None: # change goal between resets
            self.goal =np.random.uniform(low=[-0.4,-0.4,-0.3],high=[0.4,0.0,-0.3]).reshape(3,1)
            self.shuffled_order = np.random.permutation([0, 1, 2])  # get a new shuffled order on a new task

        qpos[-7:-4] = self.goal
        qvel[-7:] = 0  # todo: I think we may want to expand this to make sure distractors are also fixed
        setattr(self.model.data, 'qpos', qpos)
        setattr(self.model.data, 'qvel', qvel)
        self.model.data.qvel = qvel
        self.model._compute_subtree()
        self.model.forward()
        self.current_com = self.model.data.com_subtree[0]
        self.dcom = np.zeros_like(self.current_com)
        return self.get_current_obs()



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
