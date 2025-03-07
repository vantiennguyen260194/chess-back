import pickle
import random
import numpy as np
from collections import defaultdict


class QLearningChess:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.2, model_path="models/q_table.pkl"):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.model_path = model_path
        self.q_table = defaultdict(lambda: np.zeros(len(env.get_valid_moves())))

        self.load_model()

    def save_model(self):
        with open(self.model_path, "wb") as f:
            pickle.dump(dict(self.q_table), f)

    def load_model(self):
        try:
            with open(self.model_path, "rb") as f:
                self.q_table = pickle.load(f)
        except FileNotFoundError:
            print("Model file not found, starting from scratch.")

    def choose_action(self, state):
        valid_moves = self.env.get_valid_moves()

        if random.uniform(0, 1) < self.epsilon:
            return random.choice(valid_moves)
        else:
            q_values = self.q_table[state]
            max_q_value = np.max(q_values)
            max_q_actions = [i for i, q in enumerate(q_values) if q == max_q_value]
            return valid_moves[random.choice(max_q_actions)]

    def update_q_table(self, state, action, reward, next_state):
        valid_moves = self.env.get_valid_moves()

        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(len(valid_moves))

        next_q_values = self.q_table[next_state]
        max_next_q = np.max(next_q_values)
        action_idx = valid_moves.index(action)
        
        # Q-learning update rule
        current_q = self.q_table[state][action_idx]
        self.q_table[state][action_idx] += self.alpha * (reward + self.gamma * max_next_q - current_q)
    
    def train(self, episodes=1000):
        for episode in range(episodes):
            state = self.env.reset()
            done = False

            while not done:
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)
                self.update_q_table(state, action, reward, next_state)
                state = next_state
                print(f"Episode {episode+1}/{episodes}, Reward: {reward}")
