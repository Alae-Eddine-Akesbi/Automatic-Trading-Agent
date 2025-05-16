# environment.py
class TradingEnvironment:

    TRANSACTION_COST = 0.0005

    def __init__(self, data):
        self.data = data
        self.n_steps = len(data)
        self.reset()

    def reset(self):
        self.current_step = 0
        self.position = 0
        self.buy_price = 0.0
        self.total_profit = 0.0
        self.history = []      
        self.profits = []      
        return self._get_state()

    def _get_state(self):
        row = self.data.iloc[self.current_step]
        return tuple((
            int(row['Return_Class']),
            int(row['MA_Signal']),
            int(self.position),
            int(row['RSI'] > 70),
            int(row['RSI'] < 30),
            int(row['Adj Close'] > row['Bollinger_Upper']),
            int(row['Adj Close'] < row['Bollinger_Lower']),
            int(row['MACD'] > 0),
            int(row['Volume_Norm'] > 1),
        ))

    def step(self, action):
        done = False
        reward = 0.0
        price = float(self.data.iloc[self.current_step]['Adj Close'])

        if action == 1 and self.position == 0:
            self.position = 1
            self.buy_price = price * (1 + self.TRANSACTION_COST)
            self.history.append((self.current_step, 'BUY', price))

        elif action == 2 and self.position == 1:
            sell_price = price * (1 - self.TRANSACTION_COST)
            profit = float(sell_price - self.buy_price)  # ✅ correction
            reward = profit
            self.total_profit += profit
            self.profits.append(profit)
            self.position = 0
            self.history.append((self.current_step, 'SELL', price))

        self.current_step += 1
        if self.current_step >= self.n_steps:
            next_state = None
            done = True
        else:
            next_state = self._get_state()

        return next_state, reward, done
