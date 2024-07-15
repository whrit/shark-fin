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
        agent.load_state_dict(torch.load(f'agent_dqn_{ticker}.pt'))
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
    
    # Trading environment
    try:
        Env = portfolio_tradeEnv(day=0, balance=100000, stock=trade_data, cost=0.003)
        print("Trading environment initialized")
    except Exception as e:
        print(f"Error initializing trading environment: {str(e)}")
        return
    
    return_List = []
    action_List = []
    portfolio_value_List = []
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
    
    print("Starting trading loop")
    step = 0
    while not done:
        try:
            action = agent.take_action(state, random=False)
            next_state, reward, done, _ = Env.step(action)
            next_state = torch.tensor(Normalize(next_state.values), dtype=torch.float32).reshape(1, -1).to(device="cuda:0")
            state = next_state
            return_List.append(reward)
            action_List.append(action)
            portfolio_value = Env.balance + Env.shares[-1] * Env.stock_state['Close'].values[-1]
            portfolio_value_List.append(portfolio_value)
            
            step += 1
            if step % 50 == 0:
                print(f"Step {step}: Action = {action}, Reward = {reward:.4f}, Portfolio Value = {portfolio_value:.2f}")
            
            if done:
                print(f"Episode finished after {step} steps")
                break
        except Exception as e:
            print(f"Error during trading loop at step {step}: {str(e)}")
            break
    
    print(f"Trading loop completed. Total steps: {step}")
    print(f"Final portfolio value: {portfolio_value_List[-1]:.2f}")
    
    # Visualize reward
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(return_List)), return_List)
    plt.title(f"Rewards over time for {ticker}")
    plt.xlabel("Steps")
    plt.ylabel("Reward")
    plt.savefig(f"{ticker}_rewards.png")
    plt.close()
    
    # Visualize portfolio value
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(portfolio_value_List)), portfolio_value_List)
    plt.title(f"Portfolio Value over time for {ticker}")
    plt.xlabel("Steps")
    plt.ylabel("Portfolio Value")
    plt.savefig(f"{ticker}_portfolio_value.png")
    plt.close()
    
    # Visualize actions
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(action_List)), action_List)
    plt.title(f"Actions over time for {ticker}")
    plt.xlabel("Steps")
    plt.ylabel("Action")
    plt.yticks([-1, 0, 1], ['Sell', 'Hold', 'Buy'])
    plt.savefig(f"{ticker}_actions.png")
    plt.close()

    print(f"Plots saved as {ticker}_rewards.png, {ticker}_portfolio_value.png, and {ticker}_actions.png")

if __name__ == '__main__':
    DQN_trade(ticker='SPY')