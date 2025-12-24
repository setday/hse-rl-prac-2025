from collections import defaultdict

import numpy as np


class QLearningAgent:
    """Custom Q-learning agent implementation based on prac3."""
    
    def __init__(self, action_size, alpha=0.1, gamma=0.99, epsilon=0.05):
        """
        Initialize Q-learning agent.
        
        Args:
            state_size: Size of the observation/state space
            action_size: Number of possible actions
            alpha: Learning rate
            gamma: Discount factor
            epsilon: Epsilon for epsilon-greedy exploration
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.action_size = action_size
        
        # Initialize Q-table with zeros
        self.Q = defaultdict(lambda: np.zeros(action_size))
    
    def select_action_eps_greedy(self, state):
        """
        Select action using epsilon-greedy strategy.
        With probability epsilon, select random action.
        Otherwise, select greedy action (argmax Q).
        """
        if state not in self.Q or np.random.rand() < self.epsilon:
            action = np.random.randint(self.action_size)
        else:
            action = np.argmax(self.Q[state])
        return action
    
    def update_Q(self, state, action, reward, next_state, done):
        """
        Update Q-table using Q-learning update rule:
        Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))
        """
        # Compute V(next_state) - estimate of optimal future value
        if next_state not in self.Q or done:
            V_next_s = 0  # Terminal state has value 0
        else:
            V_next_s = np.max(self.Q[next_state])
        
        # Compute TD error
        td_error = reward + self.gamma * V_next_s - self.Q[state][action]
        
        # Update Q-function
        self.Q[state][action] += self.alpha * td_error
    
    def act(self, state):
        """Select action for given state using epsilon-greedy strategy."""
        return self.select_action_eps_greedy(state)
    
    def learn(self, state, action, reward, next_state, done):
        """Learn from experience."""
        self.update_Q(state, action, reward, next_state, done)
