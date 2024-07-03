
import torch
import matplotlib.pyplot as plt
from Data.Stock_data import data
from tradeEnv import portfolio_tradeEnv
from Model.Deep_Q_Network import Q_Net, DQN_Agent


def Normalize(state):
    # 状态标准化
    state = (state - state.mean())/(state.std())
    return state


def DQN_trade(ticker):
    """

    :param ticker: 交易股票代码
    :return:
    """
    # 智能体
    agent = DQN_Agent(state_dim=150, hidden_dim=30, action_dim=3, lr=0.001, device="cuda:0", gamma=0.95,
                      epsilon=0.01, target_update=10)
    agent.load_state_dict(torch.load(f'../Result/agent_dqn_{ticker}.pt'))
    # 交易数据
    trade_df = data(ticker=ticker, window_length=15, t=2000).trade_data()
    # 交易环境
    Env = portfolio_tradeEnv(day=0, balance=1, stock=trade_df, cost=0.003)
    return_List = []
    done = False
    episode_return = 0
    # 返回初始状态
    state = Env.reset()
    # 标准滑state
    state = torch.tensor(Normalize(state).values, dtype=torch.float32).reshape(1, -1).to(device="cuda:0")
    while not done:
        action = agent.take_action(state, random=False)
        next_state, reward, done, _ = Env.step(action)
        next_state = torch.tensor(Normalize(next_state).values, dtype=torch.float32).reshape(1, -1).to(device="cuda:0")
        state = next_state
        return_List.append(reward)
    # 可视化reward
    plt.plot(range(len(return_List)), return_List)
    plt.show()


if __name__ == '__main__':
    DQN_trade(ticker='SPY')
