from sandbox.rocky.tf.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import FiniteDifferenceHvp
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.envs.mujoco.pusher_env import PusherEnv

#from gym.envs.mujoco.pusher import PusherEnv
from rllab.envs.gym_env import GymEnv

def run_task(*_):

    env = TfEnv(normalize(GymEnv("Reacher-v1", force_reset=True, record_video=True)))
    #env = TfEnv(normalize(PusherEnv()))
    policy = GaussianMLPPolicy(
        name="policy",
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(128, 128)
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=100*500,
        max_path_length=100,
        n_itr=200,
        discount=0.99,
        step_size=0.01,
        force_batch_sampler=True,
        # optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5))
    )
    algo.train()

run_experiment_lite(
    run_task,
    # Number of parallel workers for sampling
    n_parallel=4,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="last",
    exp_prefix='trpo_push_teddy',
    python_command='python3',
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used
    seed=1,
    # plot=True,
)
