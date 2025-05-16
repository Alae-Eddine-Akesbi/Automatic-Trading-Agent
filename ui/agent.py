import numpy as np

class QLearningAgent:
    def __init__(self, action_size=3, alpha=0.1, gamma=0.95,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.q_table = {}
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def _ensure_state_exists(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_size)

    def get_qs(self, state):
        self._ensure_state_exists(state)
        return self.q_table[state]

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        return int(np.argmax(self.get_qs(state)))

    def learn(self, state, action, reward, next_state):
        self._ensure_state_exists(state)
        old_q = self.q_table[state][action]
        future_q = 0.0 if next_state is None else np.max(self.get_qs(next_state))
        self.q_table[state][action] = old_q + self.alpha * (reward + self.gamma * future_q - old_q)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
