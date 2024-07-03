# Quantitative-Trading
Single stock trading strategies based on DQN, DDQN, Behavioural cloning and BCDDQN. 

## Technology Stack

- Pytorch
- Request

## Result

![Project Screenshot](https://github.com/1998-Chen/Quantitative-Trading/blob/main/Result/000001_SZ.svg)

## Code examples
#### Trading Logic
        def make_action(self, action, begin_value):
        if self.shares[-1] == 0:
            # No position
            if action == 1:  # Buy
                if self.balance > 0:
                    # Buy all
                    self.shares[-1] = (1 - self.transaction_cost) * self.balance / self.stock[self.day-1].close.values[-1]
                    self.balance = 0
                    self.reward = np.log(self.balance + self.stock[self.day].close.values[-1] * self.shares[-1] - begin_value)
            if action == 0:
                # Risk-free profit
                self.reward = 0.0015
            if action == -1:
                self.reward = 0
        else:
            # Full position
            if action == 1:
                self.reward = (self.stock[self.day-1].close[-1] / self.stock[self.day].close[-1])-1 
            if action == 0:
                # Hold: price change
                self.reward = (self.stock[self.day-1].close[-1] / self.stock[self.day].close[-1])-1 
            if action == -1:
                if self.shares[-1] > 0:
                    # Sell
                    cash = self.stock[self.day-1].close.values[-1] * self.shares[-1] * (1 - self.transaction_cost)
                    self.balance += cash
                    # Update shares
                    self.shares[-1] = 0
                    self.reward = np.log(self.balance + self.stock[self.day].close.values[-1] * self.shares[-1] - begin_value)

## Copyright information

If you like this project, please cite "Deep reinforcement learning stock trading strategy considering behavioral cloning" from the Journal of Systems Management.

## Corresponding author

zhangy@gdut.edu.cn

## Support

This research was supported by the Guangdong Basic and Applied Basic Research Foundation (No. 2023A1515012840)
