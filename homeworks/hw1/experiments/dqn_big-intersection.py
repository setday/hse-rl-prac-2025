from sumo_rl import SumoEnvironment
from stable_baselines3.dqn.dqn import DQN

from dqn_strategy import PrioritizedDQNAgent



EPISODE_LENGTH = 5400  # seconds
REPLAY_BUFFER_SIZE = 50000
LEARNING_TIMESTEPS = 20000 # reduced for quicker experiments (100000 -> 20000)

env = SumoEnvironment(
    net_file="../big-intersection/big-intersection.net.xml",
    single_agent=True,
    route_file="../big-intersection/routes.rou.xml",
    out_csv_name="outputs/test/dqn-prioritized/",
    use_gui=False,
    num_seconds=EPISODE_LENGTH,
    yellow_time=4,
    min_green=5,
    max_green=60,
)

# Create prioritized agent
agent_prioritized = PrioritizedDQNAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n,
    hidden_dims=(256, 256),
    lr=1e-3,
    gamma=0.99,
    eps_start=0.4,
    eps_end=0.02,
    replay_buffer_size=REPLAY_BUFFER_SIZE,
    batch_size=32
)
agent_prioritized.learn(env, total_timesteps=LEARNING_TIMESTEPS, eval_schedule=EPISODE_LENGTH * 5)

print("Prioritized DQN training finished.")

env.close()

env = SumoEnvironment(
    net_file="../big-intersection/big-intersection.net.xml",
    single_agent=True,
    route_file="../big-intersection/routes.rou.xml",
    out_csv_name="outputs/test/dqn-builtin/",
    use_gui=False,
    num_seconds=EPISODE_LENGTH,
    yellow_time=4,
    min_green=5,
    max_green=60,
)

model = DQN(
    env=env,
    policy="MlpPolicy",
    learning_rate=1e-3,
    learning_starts=0,
    buffer_size=REPLAY_BUFFER_SIZE,
    train_freq=1,
    target_update_interval=500,
    exploration_fraction=0.05,
    exploration_final_eps=0.01,
    verbose=1,
)
model.learn(total_timesteps=LEARNING_TIMESTEPS)

print("Builtin training finished.")
