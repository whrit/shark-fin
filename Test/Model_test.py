import torch
import matplotlib.pyplot as plt
from Data.Stock_data import data
from tradeEnv import portfolio_tradeEnv
from Model.Deep_Q_Network import Q_Net, DQN_Agent
import numpy as np

def Normalize(state):
    return (state - state.mean()) / (state.std())

def DQN_trade(ticker):
    print(f"Starting DQN_trade for ticker: {ticker}")
    
    # Agent
    try:
        agent = DQN_Agent(state_dim=150, hidden_dim=30, action_dim=3, lr=0.001, device="cuda:0", gamma=0.95,
                          epsilon=0.01, target_update=10)
        agent.load_state_dict(torch.load(f'Result/agent_dqn_{ticker}.pt'))
        print("Agent loaded successfully")
    except Exception as e:
        print(f"Error loading agent: {str(e)}")
        return
    
    # Trading data
    try:
        data_obj = data(ticker=ticker, window_length=15, t=2000)
        trade_data = data_obj.trade_data()
        print(f"Trade data retrieved: {len(trade_data)} data points")
    except Exception as e:
        print(f"Error retrieving trade data: {str(e)}")
        return
    
    if not trade_data:
        print(f"Error: No data available for ticker {ticker}")
        return
    
    print(f"Debug: trade_data length: {len(trade_data)}")
    print(f"Debug: Sample data point shape: {trade_data[0].shape}")
    print(f"Debug: Sample data point:\n{trade_data[0]}")
    
    # Trading environment
    try:
        Env = portfolio_tradeEnv(day=0, balance=1, stock=trade_data, cost=0.003)
        print("Trading environment initialized")
    except Exception as e:
        print(f"Error initializing trading environment: {str(e)}")
        return
    
    return_List = []
    done = False
    
    # Get initial state
    try:
        state = Env.reset()
        print("Environment reset successful")
    except Exception as e:
        print(f"Error resetting environment: {str(e)}")
        return
    
    if state is None:
        print("Error: Failed to initialize environment state")
        return
    
    # Normalize state
    try:
        state = torch.tensor(Normalize(state.values), dtype=torch.float32).reshape(1, -1).to(device="cuda:0")
        print("Initial state normalized")
    except Exception as e:
        print(f"Error normalizing initial state: {str(e)}")
        return
    
    while not done:
        try:
            action = agent.take_action(state, random=False)
            next_state, reward, done, _ = Env.step(action)
            next_state = torch.tensor(Normalize(next_state.values), dtype=torch.float32).reshape(1, -1).to(device="cuda:0")
            state = next_state
            return_List.append(reward)
        except Exception as e:
            print(f"Error during trading loop: {str(e)}")
            break
    
    # Visualize reward
    plt.plot(range(len(return_List)), return_List)
    plt.title(f"Rewards over time for {ticker}")
    plt.xlabel("Steps")
    plt.ylabel("Reward")
    plt.show()

if __name__ == '__main__':
    DQN_trade(ticker='SPY')