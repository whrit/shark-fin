import numpy as np

class portfolio_tradeEnv:
    # action = [-1, 0, 1]
    # all balance 

    def __init__(self, day, stock, balance, cost) -> None:
        self.day = day
        self.stock = stock  # data
        self.stock_state = self.stock[self.day]
        self.balance = balance
        self.shares = [0] * 1
        self.transaction_cost = cost
        self.terminal = False
        self.rate = []
        self.reward = 0

    def step(self, action):
        self.terminal = self.day >= len(self.stock) - 1
        if self.terminal:
            # print('Balance:', self.balance, 'Close Price:', self.stock_state.Close.values[-1], 'Shares:', self.shares[-1])
            return self.stock_state, self.reward, self.terminal, {}

        else:
            begin_assert_value = self.balance + self.stock_state.Close.values[-1] * self.shares[-1]
            if action == -1:
                # Execute sell action
                self.sell(action)
            if action == 0:
                # Execute hold action, i.e., do not perform any trading action
                self.hold(action)
            if action == 1:
                # Execute buy action
                self.buy(action)

            self.day += 1
            # print('Day:', self.day)
            self.stock_state = self.stock[self.day]
            end_assert_value = self.balance + self.stock_state.Close.values[-1] * self.shares[-1]
            self.rate.append((end_assert_value - 100000) / 100000 + 1)
            self.reward = (end_assert_value - begin_assert_value) / begin_assert_value
            # print('Day:', self.day, 'Balance:', self.balance, 'Close Price:', self.stock_state.Close.values[-1],
            #       'Shares:', self.shares[-1], 'Value:', end_assert_value, 'Action:', action)

            return self.stock_state, self.reward, self.terminal, {}

    def buy(self, action):
        # Check if the account balance supports the buy action

        if self.balance > 0:
            self.shares.append((1 - self.transaction_cost) * self.balance / self.stock_state.Close.values[-1])
            self.balance = 0
            # print('Buy Share:', action * self.HMAX_SHARE)
        else:
            pass

    def hold(self, action):
        pass

    def sell(self, action):
        # Sell all shares
        cash = self.stock_state.Close.values[-1] * self.shares[-1] * (1 - self.transaction_cost)
        self.balance += cash
        # Update shares
        self.shares.append(0)

    def reset(self, ):
        self.day = 0
        self.balance = 100000
        self.stock_state = self.stock[self.day]
        self.terminal = False
        return self.stock_state
