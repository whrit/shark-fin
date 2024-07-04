# Model_train.py

import torch
import matplotlib.pyplot as plt
from Data.Stock_data import data
from tradeEnv import portfolio_tradeEnv
from Model.Deep_Q_Network import Q_Net, DQN_Agent
from tqdm import tqdm
from colorama import Fore, Style

def Normalize(state):
    # State normalization
    state = (state - state.mean()) / (state.std())
    return state

def DQN_train(episode, ticker, minimum):
    """
    :param episode: Number of training episodes
    :param ticker: Stock ticker for training
    :param minimum: Minimum size of experience replay buffer
    :return:
    """
    # Agent
    agent = DQN_Agent(state_dim=150, hidden_dim=30, action_dim=3, lr=0.001, device="cuda:0", gamma=0.95,
                      epsilon=0.01, target_update=10)
    # Training data
    train_df = data(ticker=ticker, window_length=15, t=2000).train_data()
    # Training environment
    Env = portfolio_tradeEnv(day=0, balance=1, stock=train_df, cost=0.003)
    return_List = []
    # Agent experience buffer
    transition_dict = {
        'states': [],
        'actions': [],
        'next_states': [],
        'rewards': [],
        'dones': [],
    }
    for i in tqdm(range(episode), desc=Fore.GREEN + 'Training Episodes' + Style.RESET_ALL, ncols=100, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', leave=True):
        done = False
        episode_return = 0
        # Return initial state
        state = Env.reset()
        # Standardize state
        state = torch.tensor(Normalize(state).values, dtype=torch.float32).reshape(1, -1).to(device="cuda:0")
        while not done:
            action = agent.take_action(state, random=True)
            next_state, reward, done, _ = Env.step(action)
            # Experience replay buffer
            transition_dict['states'].append(state)
            transition_dict['actions'].append(action + 1)
            transition_dict['rewards'].append(reward)
            transition_dict['dones'].append(done)
            next_state = torch.tensor(Normalize(next_state).values, dtype=torch.float32).reshape(1, -1).to(device="cuda:0")
            transition_dict['next_states'].append(next_state)
            state = next_state
            episode_return += reward
            if len(transition_dict['states']) >= minimum:
                # Agent learning
                # print('---Learning---')
                agent.update(transition_dict)

        return_List.append(episode_return)
    # Model saving
    PATH = f'agent_dqn_{ticker}.pt'
    torch.save(agent.state_dict(), PATH, _use_new_zipfile_serialization=False)
    # Visualize reward
    plt.plot(range(len(return_List)), return_List)
    plt.show()

if __name__ == '__main__':
    import time

    # Improvement: Periodically clear the experience buffer and sample a fixed number of experience tuples
    start = time.time()
    DQN_train(episode=100, ticker='SPY', minimum=1500)
    end = time.time()
    print('Training time', end - start)
