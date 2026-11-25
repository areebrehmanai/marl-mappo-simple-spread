"""Compare the trained MAPPO agents with two simple baselines over 100 episodes."""
import numpy as np
import torch

from mappo import Actor, CentralCritic, play, simple_spread

EPISODES = 100
env = simple_spread.parallel_env(N=3, max_cycles=25, continuous_actions=False)
env.reset(seed=123)
agents = list(env.possible_agents)
eye = np.eye(len(agents), dtype=np.float32)
obs_dim = env.observation_space(agents[0]).shape[0]
n_actions = env.action_space(agents[0]).n


def baseline(choose_action):
    totals = []
    for _ in range(EPISODES):
        env.reset()
        total = 0.0
        while env.agents:
            _, rew, _, _, _ = env.step({a: choose_action(a) for a in env.agents})
            total += np.mean(list(rew.values()))
        totals.append(total)
    return np.mean(totals)


ckpt = torch.load("results/mappo_simple_spread.pt")
actor = Actor(obs_dim + len(agents), n_actions)
actor.load_state_dict(ckpt["actor"])
critic = CentralCritic(obs_dim * len(agents))

print(f"Random actions      : {baseline(lambda a: env.action_space(a).sample()):7.2f}")
print(f"Agents stand still  : {baseline(lambda a: 0):7.2f}")
trained = np.mean([sum(play(env, actor, critic, agents, eye)['rew']) for _ in range(EPISODES)])
print(f"Trained MAPPO       : {trained:7.2f}")
print("(Mean team return over 100 episodes; higher = better)")
