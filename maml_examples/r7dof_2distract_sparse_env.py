import numpy as np
from rllab.envs.mujoco import mujoco_env
from rllab.misc.overrides import overrides
from rllab.envs.base import Step
import copy
import joblib
from rllab.sampler.utils import rollout, joblib_dump_safe


from rllab.core.serializable import Serializable


class Reacher7Dof2DistractSparseEnv(
    mujoco_env.MujocoEnv, Serializable
):
    def __init__(self, distance_metric_order=None, *args, **kwargs):
        self.goal = None
        self.shuffle_order = None
        self.objects = ["goal","distract1","distract2"]
        self.__class__.FILE = 'r7dof_versions/reacher_7dof_2distract.xml'
        seed = kwargs['envseed']
        super().__init__(envseed=seed)
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
            self.get_body_com(self.objects[int(self.shuffle_order[0][0])]),
            self.get_body_com(self.objects[int(self.shuffle_order[1][0])]),
            self.get_body_com(self.objects[int(self.shuffle_order[2][0])]),
        ])

    def step(self, action):
        self.frame_skip = 5
        distance = np.linalg.norm(
            self.get_body_com("tips_arm") - self.get_body_com("goal")
        )
        # reward = - distance
        reward = 1.0 if distance < 0.2 else 0.0
        self.forward_dynamics(action)
        # self.do_simulation(action, self.frame_skip)
        next_obs = self.get_current_obs()
        done = False
        return Step(next_obs, reward, done) #, dict(distance=distance)

    def sample_goals(self, num_goals):
        goals_list = []
        for _ in range(num_goals):
            newgoal = np.random.uniform(low=[-0.4,-0.4,-0.3],high=[0.4,0.0,-0.3]).reshape(3,1)
            distract1 = np.random.uniform(low=[-0.4,-0.4,-0.3],high=[0.4,0.0,-0.3]).reshape(3,1)
            distract2 = np.random.uniform(low=[-0.4,-0.4,-0.3],high=[0.4,0.0,-0.3]).reshape(3,1)
            shuffle_order = np.random.permutation([0,1,2]).reshape(3,1)
            goals_list.append([newgoal, distract1, distract2, shuffle_order])
        return np.array(goals_list)

    def enrich_goals_pool(self, goals_pool):
        new_goals_list = []
        goals_list = goals_pool['goals_pool']
        new_goals_pool = {}
        for goal in goals_list:
            distract1 = np.random.uniform(low=[-0.4,-0.4,-0.3],high=[0.4,0.0,-0.3]).reshape(3,1)
            distract2 = np.random.uniform(low=[-0.4,-0.4,-0.3],high=[0.4,0.0,-0.3]).reshape(3,1)
            shuffle_order = np.random.permutation([0,1,2]).reshape(3,1)
            new_goals_list.append([goal, distract1, distract2, shuffle_order])
        new_goals_pool['goals_pool'] = np.array(new_goals_list)
        new_goals_pool['idxs_dict'] = goals_pool['idxs_dict']
        return new_goals_pool

    def enrich_expert_trajectories(self, goal_folder, goal_number):
        trajs_for_goal = joblib.load(goal_folder+str(goal_number) + ".pkl")
        goal_distractions_and_shuffle_order = joblib.load(goal_folder+"goals_pool.pkl")['goals_pool'][goal_number]
        shuffle_order = goal_distractions_and_shuffle_order[-1].reshape((3,))
        goal_and_distractions = np.concatenate((
            goal_distractions_and_shuffle_order[int(shuffle_order[0])],
            goal_distractions_and_shuffle_order[int(shuffle_order[1])],
            goal_distractions_and_shuffle_order[int(shuffle_order[2])]
        ))
        print("goals for index", goal_number, ":\n", goal_and_distractions)
        new_trajs_for_goal = []
        for traj in trajs_for_goal:
            obs = traj['observations']
            new_obs = [np.concatenate((obs_step,goal_and_distractions.reshape((9,)))) for obs_step in obs]
            new_traj = copy.deepcopy(traj)
            new_traj['observations'] = new_obs
            new_trajs_for_goal.append(new_traj)
        joblib_dump_safe(new_trajs_for_goal,goal_folder+str(goal_number)+"dist.pkl")




    @overrides
    def reset(self, reset_args=None, **kwargs):



        qpos = np.copy(self.init_qpos)
        qvel = np.copy(self.init_qvel) + 0.0* self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )

        goal = reset_args
        if goal is not None:
            self.goal = goal[0]
            self.distract1 = goal[1]
            self.distract2 = goal[2]
            self.shuffle_order = goal[3]
        # elif self.goal is None: # do not change goal between resets, only at initialization and when explicitly given a new goal
        elif reset_args is None: # change goal between resets
            self.goal =np.random.uniform(low=[-0.4,-0.4,-0.3],high=[0.4,0.0,-0.3]).reshape(3,1)
            self.distract1 = np.random.uniform(low=[-0.4, -0.4, -0.3], high=[0.4, 0.0, -0.3]).reshape(3, 1)
            self.distract2 = np.random.uniform(low=[-0.4, -0.4, -0.3], high=[0.4, 0.0, -0.3]).reshape(3, 1)
            self.shuffle_order = np.random.permutation([0, 1, 2]).reshape(3,1)  # get a new shuffled order on a new task


        qpos[-7:-4] = self.goal
        qpos[-14:-11] = self.distract1
        qpos[-21:-18] = self.distract2
        qvel[-7:] = 0  # todo: I think we may want to expand this to make sure distractors are also fixed
        setattr(self.model.data, 'qpos', qpos)
        setattr(self.model.data, 'qvel', qvel)
        self.model.data.qvel = qvel
        self.model._compute_subtree()
        self.model.forward()
        self.current_com = self.model.data.com_subtree[0]
        self.dcom = np.zeros_like(self.current_com)
        return self.get_current_obs()


