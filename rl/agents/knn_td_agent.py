import math
import random
import numpy as np
from copy import deepcopy

from operator import itemgetter


def euclidean_distance(x, y):
    return math.sqrt(sum([abs(x[k] - y[k]) ** 2 for k in range(len(x))]))


def hamming_distance(x, y):
	d = 0
	for i, j in zip(x, y):
		if i != j:
			d += 1
	return d


class KNNTDAgent:

    def __init__(self, starting_state, state_space, action_space, k, alpha=0.5, gamma=0.95, initial_epsilon=1.0, min_epsilon=0.0, decay_episodes=20):
        self.state = tuple(starting_state)
        self.state_space = state_space
        self.action_space = action_space
        self.action = None
        self.k = k
        self.alpha = alpha
        self.gamma = gamma

        self.epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.decay = (initial_epsilon - min_epsilon) / decay_episodes
        self.episode = 0

        # experiences of the agent
        self.observations = {self.state: [0 for _ in range(self.action_space.n)]}

        self.expected_q_values = [0 for _ in range(self.action_space.n)]
        self.nearest_neighbors = []
        self.probabilities = []

        self.previous_state_expected_q_values = [0 for _ in range(self.action_space.n)]
        self.previous_state_nearest_neighbors = []
        self.previous_state_probabilities = []

    def estimate_expected_values(self):
        # estimates q values of current state
        self.expected_q_values = [0 for _ in range(self.action_space.n)]
        self.nearest_neighbors = []
        weights = []
        self.probabilities = []
        
        # orders all experiences by distance
        ord_distances = sorted([(hamming_distance(self.state, state), state) for state in self.observations.keys()],
                               key=itemgetter(0))
        """
        ord_distances = sorted([(euclidean_distance(self.state, state), state) for state in self.observations.keys()],
                               key=itemgetter(0))
        """

        # selects the k closest experiences
        if len(ord_distances) > self.k:
            ord_distances = ord_distances[:self.k]

        # computes weights of experiences
        for d, state in ord_distances:
            self.nearest_neighbors.append(state)
            weights.append(1 / (1 + d ** 2))

        # computes probabilities of experiences
        sum_weights = sum(weights)
        for w in weights:
            self.probabilities.append(w/sum_weights)

        # computes expected q values of current state
        for p, state in zip(self.probabilities, self.nearest_neighbors):
            for a in range(self.action_space.n):
                self.expected_q_values[a] += p * self.observations[state][a]

        self.observations[self.state] = deepcopy(self.expected_q_values)

    def act(self, action_masks, episode):
        legal_actions = [action for action, is_valid in enumerate(action_masks) if is_valid]
        if np.random.rand() > self.epsilon:
            # self.action = np.argmax(self.expected_q_values)
            legal_actions_index = np.argmax([self.expected_q_values[legal] for legal in legal_actions])
            self.action = legal_actions[legal_actions_index]
        else:
            # self.action = int(self.action_space.sample())
             self.action = random.choice(legal_actions)

        # decays value of epsilon
        if self.episode != episode:
            self.episode = episode
            self.epsilon = max(self.epsilon - self.decay, self.min_epsilon)
            # self.epsilon = max(self.epsilon * self.decay, self.min_epsilon)

        return self.action

    def learn(self, next_state, reward, done=False):
        self.state = tuple(next_state)
        if self.state not in self.observations:
            self.observations[self.state] = [0 for _ in range(self.action_space.n)]
        self.estimate_expected_values()

        # computes td error and updates q values of nearest neighbors
        td_error = reward + self.gamma * max(self.expected_q_values) - self.previous_state_expected_q_values[self.action]
        for p, state in zip(self.previous_state_probabilities, self.previous_state_nearest_neighbors):
            self.observations[state][self.action] += self.alpha * td_error * p

        self.previous_state_expected_q_values = deepcopy(self.expected_q_values)
        self.previous_state_nearest_neighbors = deepcopy(self.nearest_neighbors)
        self.previous_state_probabilities = deepcopy(self.probabilities)
