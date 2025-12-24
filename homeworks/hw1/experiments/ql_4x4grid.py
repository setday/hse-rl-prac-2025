from sumo_rl import SumoEnvironment
from sumo_rl.agents import QLAgent
from sumo_rl.exploration import EpsilonGreedy
from ql_strategy import QLearningAgent
from tqdm.auto import tqdm


sumo_rl_path = "~/AppData/Local/Programs/Python/Python312/Lib/site-packages/sumo_rl"

EPISODE_LENGTH = 4000  # seconds
LEARNING_TIMESTEPS = 100000 # reduced for quicker experiments (480000 (30 * 4 * 4000) -> 100000)

alpha = 0.1
gamma = 0.99
epsilon = 0.05

# env = SumoEnvironment(
#     net_file=f"{sumo_rl_path}/nets/4x4-Lucas/4x4.net.xml",
#     route_file=f"{sumo_rl_path}/nets/4x4-Lucas/4x4c1c2c1c2.rou.xml",
#     out_csv_name="outputs/test/ql-builtin/",
#     use_gui=False,
#     num_seconds=EPISODE_LENGTH,
#     min_green=5,
#     delta_time=5,
# )

# print("Running with custom QLAgent...")

# initial_states = env.reset()
# ql_agents = {
#     ts: QLAgent(
#         starting_state=env.encode(initial_states[ts], ts),
#         state_space=env.observation_space,
#         action_space=env.action_space,
#         alpha=alpha,
#         gamma=gamma,
#         exploration_strategy=EpsilonGreedy(initial_epsilon=epsilon, min_epsilon=0.005, decay=1),
#     )
#     for ts in env.ts_ids
# }

# for episode in tqdm(range(LEARNING_TIMESTEPS//EPISODE_LENGTH)):
#     if episode != 0:
#         initial_states = env.reset()
#         for ts in initial_states.keys():
#             ql_agents[ts].state = env.encode(initial_states[ts], ts)

#     infos = []
#     done = {"__all__": False}
#     while not done["__all__"]:
#         actions = {ts: ql_agents[ts].act() for ts in ql_agents.keys()}

#         s, r, done, info = env.step(action=actions)

#         for agent_id in s.keys():
#             ql_agents[agent_id].learn(next_state=env.encode(s[agent_id], agent_id), reward=r[agent_id])

# env.close()

env = SumoEnvironment(
    net_file=f"{sumo_rl_path}/nets/4x4-Lucas/4x4.net.xml",
    route_file=f"{sumo_rl_path}/nets/4x4-Lucas/4x4c1c2c1c2.rou.xml",
    out_csv_name="outputs/test/ql-custom/",
    use_gui=False,
    num_seconds=EPISODE_LENGTH,
    min_green=5,
    delta_time=5,
)

print("Running with custom QLearningAgent...")

initial_states = env.reset()

# Initialize custom agents
custom_agents = {
    ts: QLearningAgent(
        # state_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon
    )
    for ts in env.ts_ids
}

for episode in tqdm(range(LEARNING_TIMESTEPS//EPISODE_LENGTH)):
    if episode != 0:
        initial_states = env.reset()

    done = {"__all__": False}
    while not done["__all__"]:
        actions = {ts: custom_agents[ts].act(env.encode(initial_states[ts], ts)) for ts in custom_agents.keys()}

        s, r, done, info = env.step(action=actions)

        for agent_id in s.keys():
            next_state_encoded = env.encode(s[agent_id], agent_id)
            custom_agents[agent_id].learn(
                state=env.encode(initial_states[agent_id], agent_id),
                action=actions[agent_id],
                reward=r[agent_id],
                next_state=next_state_encoded,
                done=done[agent_id]
            )
            initial_states[agent_id] = s[agent_id]

env.close()