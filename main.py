import bar_update
import EnvironmentManager
import Agent
import DQN
import ReplayMemory
import drive

import torch
from itertools import count
from collections import namedtuple


mini_batch = 256
gamma = 0.99
eps_start = 1
eps_end = 0.01
eps_decay = 0.00001
target_update = 10000
memory_size = 1000000
lr = 0.00001

Experience = namedtuple(
    'Experience',
    ('state', 'action', 'reward', 'next_state', 'end_state')
)

def extract_tensors(experience):
    batch = Experience(*zip(*experience))

    t1 = torch.cat(batch.state)
    t2 = torch.cat(batch.action)
    t3 = torch.cat(batch.reward).unsqueeze(dim=-1)
    t4 = torch.cat(batch.next_state)
    t5 = torch.cat(batch.end_state)
    return (t1, t2, t3, t4, t5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path = 'G:/My Drive/models/'
model_name = 'DQN_Snake.mdl'

bar = bar_update.bar_update()
drive_files = drive.model_drive(path, model_name)

model_exists = drive_files.does_exist()

policy_net = DQN.DQN(8).to(device)

if model_exists:
    checkpoint = drive_files.download()
    update = checkpoint['update']
    policy_net.load_state_dict(checkpoint['model_state_dict'])
else:
    checkpoint = None
    update = 1

target_net = DQN.DQN(8).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

em = EnvironmentManager.EnvManager(device)
strategy = Agent.EpsilonGreedyStrat(eps_start, eps_end, eps_decay)
agent = Agent.Agent(strategy, 3, device, target_net, policy_net, lr, gamma, checkpoint)
memory = ReplayMemory.ReplayMemory(memory_size)


points_all = 0
loss_all = 0
score_all = 0
High_score = 0
episode = 0


while True:
    episode += 1
    iter = 0
    loss = 0
    points = 0
    state = em.get_state()

    for timestep in count():

        action = agent.select_action(state, policy_net)
        reward, points, end_state = em.take_action(action)
        next_state = em.get_state()
        memory.push(Experience(state, action, reward, next_state, end_state))
        state = next_state

        if memory.can_provide_sample(mini_batch):
            experience = memory.sample(mini_batch)
            states, actions, rewards, next_states, mask = extract_tensors(experience)

            loss += agent.train_memory(states, actions, rewards, next_states, mask)
            iter += 1

        if em.done:
            em.reset()
            if loss != 0:
                loss_all += loss / iter
            score_all += points
            if points > High_score:
                High_score = points
            bar.print_info(episode, loss_all / episode, score_all / episode, High_score, target_update, update)
            break

    if episode % target_update == 0:
        agent.net_update()
        update += 1

        drive_files.upload(agent.policy_net, agent.optimizer, update, agent.current_step)

        bar.new_bar()

        episode = 0
        loss_all = 0
        score_all = 0
        High_score = 0