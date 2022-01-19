import torch
import EnvironmentManager
import Agent
import DQN
import Plotting
import ReplayMemory
from itertools import count
from collections import namedtuple

mini_batch = 512
gamma = 0.999
eps_start = 1
eps_end = 0.01
eps_decay = 0.00001
target_update = 5000
memory_size = 1000000
lr = 0.00005

Experience = namedtuple(
    'Experience',
    ('state', 'action', 'reward', 'next_state')
)

def extract_tensors(experience):
    batch = Experience(*zip(*experience))

    t1 = torch.cat(batch.state)
    t2 = torch.cat(batch.action)
    t3 = torch.cat(batch.reward).unsqueeze(dim=-1)
    t4 = torch.cat(batch.next_state)
    return (t1, t2, t3, t4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy_net = DQN.DQN(8).to(device)
target_net = DQN.DQN(8).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()


em = EnvironmentManager.EnvManager(device)
strategy = Agent.EpsilonGreedyStrat(eps_start, eps_end, eps_decay)
agent = Agent.Agent(strategy, 3, device, target_net, policy_net, lr, gamma)
memory = ReplayMemory.ReplayMemory(memory_size)

# Plot = Plotting.Plot()

points_all = []
loss_all = []
High_score = 0

episode = 0
loss = 0

while True:
    episode += 1
    points = 0
    em.reset()
    state = em.get_state()

    for timestep in count():
        action = agent.select_action(state, policy_net)
        reward, points = em.take_action(action)
        next_state = em.get_state()
        memory.push(Experience(state, action, reward, next_state))
        state = next_state

        if memory.can_provide_sample(mini_batch):
            experience = memory.sample(mini_batch)
            states, actions, rewards, next_states = extract_tensors(experience)

            loss += agent.train_memory(states, actions, rewards, next_states)

        if em.done:
            print(len(memory.memory))
            em.reset()
            # if points > High_score:
            #     High_score = points
            # points_all.append(points)
            # Plot.plot_points(points_all)
            print("High Score: ", High_score)
            print("Episode", episode)
            # if episode % 100 == 0:
            #     loss_all.append(loss)
            #     Plot.plot_loss(loss_all)
            #     print('Loss:', loss)
            # Plot.plot_graphs()
            break
    if episode % target_update == 0:
        agent.net_update()