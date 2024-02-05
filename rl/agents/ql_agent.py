import random
import numpy as np


class EpsilonGreedy:
    """Epsilon Greedy Exploration Strategy."""

    def __init__(self, initial_epsilon=1.0, min_epsilon=0, decay=0.995):
        """Initialize Epsilon Greedy Exploration Strategy."""
        self.initial_epsilon = initial_epsilon
        self.epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay

    def choose(self, q_table, state, action_space, action_masks):
        """Choose action based on epsilon greedy strategy."""
        legal_actions = [action for action, is_valid in enumerate(action_masks) if is_valid]
        if np.random.rand() < self.epsilon:
            action = random.choice(legal_actions)
        else:
            legal_actions_index = np.argmax([q_table[state][legal] for legal in legal_actions])
            action = legal_actions[legal_actions_index]

        self.epsilon = max(self.epsilon * self.decay, self.min_epsilon)
        # print(self.epsilon)
        return action

    def reset(self):
        """Reset epsilon to initial value."""
        self.epsilon = self.initial_epsilon


class QLAgent:
    """Q-learning Agent class."""

    def __init__(self, starting_state, state_space, action_space, alpha=0.5, gamma=0.95, exploration_strategy=EpsilonGreedy()):
        """Initialize Q-learning agent."""
        self.state = starting_state
        self.state_space = state_space
        self.action_space = action_space
        self.action = None
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = {self.state: [0 for _ in range(action_space.n)]}
        self.exploration = exploration_strategy
        self.acc_reward = 0

    def act(self, action_masks):
        """Choose action based on Q-table."""
        self.action = self.exploration.choose(self.q_table, self.state, self.action_space, action_masks)
        return self.action

    def learn(self, next_state, reward, done=False):
        """Update Q-table with new experience."""
        if next_state not in self.q_table:
            self.q_table[next_state] = [0 for _ in range(self.action_space.n)]

        s = self.state
        s1 = next_state
        a = self.action
        self.q_table[s][a] = self.q_table[s][a] + self.alpha * (
            reward + self.gamma * max(self.q_table[s1]) - self.q_table[s][a]
        )
        self.state = s1
        self.acc_reward += reward
