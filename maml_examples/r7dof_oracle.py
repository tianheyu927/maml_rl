import numpy as np
from rllab.envs.mujoco import mujoco_env
from rllab.misc.overrides import overrides



from rllab.core.serializable import Serializable


class Reacher7DofMultitaskEnvOracle(
    mujoco_env.MujocoEnv, Serializable
):
    def __init__(self, distance_metric_order=None):
        self.goal = None
        Serializable.quick_init(self, locals())
        mujoco_env.MujocoEnv.__init__(
            self,
            file_path='/home/rosen/maml_rl/vendor/mujoco_models/r7dof_versions/reacher_7dof.xml',   # You probably need to change this
            #frame_skip = 5
        )
        self._desired_xyz = np.zeros((3,1))

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 4.0

    @overrides
    def reset(self, reset_args=None):
        qpos = np.copy(self.init_qpos)
        qvel = np.copy(self.init_qvel) + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        goal_pos = reset_args
        if goal_pos is not None:
            self.goal = goal_pos
        elif self.goal is None: # do not change goal between resets
            while True:
                self.goal =np.random.uniform(low=[-0.4,-0.4,-0.3],high=[0.4,0.0,-0.3]).reshape(3,1)
                # if np.linalg.norm(self.goal) < 0.21:
                #     break

        qpos[-7:-4] = self.goal
        qvel[-7:] = 0
        setattr(self.model.data, 'qpos', qpos)
        setattr(self.model.data, 'qvel', qvel)
        self.model.data.qvel = qvel
        self.model._compute_subtree()
        self.model.forward()
        self.current_com = self.model.data.com_subtree[0]
        self.dcom = np.zeros_like(self.current_com)
        return self._get_obs()



    def reset_model(self):
        qpos = np.copy(self.init_qpos)
        qvel = np.copy(self.init_qvel) + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        print(np.shape(qpos[-7:-4]))
        qpos[-7:-4] = self._desired_xyz
        qvel[-7:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[:7],
            self.model.data.qvel.flat[:7],
            self.get_body_com("tips_arm"),
        ])

    def _step(self, a):
        distance = np.linalg.norm(
            self.get_body_com("tips_arm") - self.get_body_com("goal")
        )
        reward = - distance
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(distance=distance)

    def log_diagnostics(self, paths):
        pass
