import numpy as np
import torch
import torch.nn as nn
from collections import deque
import random


def create_network(input_dim, hidden_dims, output_dim):
    """Create a neural network for Q-function approximation."""
    network = nn.Sequential()
    for i, h in enumerate(hidden_dims):
        network.add_module(f'linear_{i}', nn.Linear(input_dim, h))
        network.add_module(f'relu_{i}', nn.ReLU())
        input_dim = h
    network.add_module(f'linear_out', nn.Linear(input_dim, output_dim))
    return network


def to_tensor(x, dtype=np.float32):
    """Convert input to PyTorch tensor."""
    if isinstance(x, torch.Tensor):
        return x
    x = np.asarray(x, dtype=dtype)
    x = torch.from_numpy(x)
    return x


def compute_td_target(Q, rewards, next_states, terminateds, gamma=0.99):
    """Compute TD target for DQN."""
    # Convert to tensors
    r = to_tensor(rewards)
    s_next = to_tensor(next_states)
    term = to_tensor(terminateds, bool)
    
    # Get Q-values for all actions in next state
    Q_sn = Q(s_next).detach()
    # Compute optimal value in next state
    V_sn = torch.max(Q_sn, dim=1)[0]
    
    # Compute TD target
    target = r + gamma * V_sn * ~term
    
    return target


def compute_td_loss(Q, states, actions, td_target, regularizer=0.1):
    """Compute TD loss for DQN."""
    # Convert to tensors
    s = to_tensor(states)
    a = to_tensor(actions, int).long()
    
    # Get Q-values for selected actions
    Q_s_a = Q(s).gather(1, a.unsqueeze(1)).squeeze(1)
    
    # Compute TD error
    td_error = td_target - Q_s_a
    
    # MSE loss
    td_losses = td_error ** 2
    loss = torch.mean(td_losses)
    # Add L1 regularization
    loss += regularizer * torch.abs(Q_s_a).mean()
    
    return loss, td_losses.detach()


class DQNAgent:
    """Deep Q-Network agent with experience replay and prioritized sampling."""
    
    def __init__(self, state_dim, action_dim, hidden_dims=(256, 256),
                 lr=1e-3, gamma=0.99, eps_start=0.4, eps_end=0.02,
                 eps_dur=25, batch_size=32, **kwargs):
        """
        Initialize DQN agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of actions
            hidden_dims: Dimensions of hidden layers
            lr: Learning rate
            gamma: Discount factor
            eps_start: Initial epsilon for exploration
            eps_end: Final epsilon for exploration
            eps_dur: Duration (in fraction of total steps) for epsilon decay
            batch_size: Batch size for training
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_dur = eps_dur
        self.batch_size = batch_size
        
        # Create Q-network
        self.Q = create_network(state_dim, hidden_dims, action_dim)
        self.optimizer = torch.optim.AdamW(self.Q.parameters(), lr=lr)
    
    def select_action(self, state, epsilon):
        """Select action using epsilon-greedy strategy."""
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        
        if random.random() < epsilon:
            action = random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                Q_s = self.Q(state)
                action = torch.argmax(Q_s).item()
        
        return action
    
    def train(self, s, a, r, s_next, terminated):
        """Perform one training step."""

        self.optimizer.zero_grad()
        td_target = compute_td_target(self.Q, [r], [s_next], [terminated], gamma=self.gamma)
        loss, _ = compute_td_loss(self.Q, [s], [a], td_target)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def save_to_buffer(self, state, action, reward, next_state, terminated, truncated):
        """Store transition in replay buffer."""
        pass  # To be implemented in subclasses if needed
    
    def learn(self, env, total_timesteps=10000, train_schedule=1, eval_schedule=1000):
        s, _ = env.reset()
        done = False

        eval_return_history = deque(maxlen=10)

        for global_step in range(1, total_timesteps + 1):
            print(f'Global step: {global_step}', end='\r')
            
            duration = self.eps_dur * total_timesteps
            if global_step >= duration:
                epsilon = self.eps_end
            else:
                epsilon = self.eps_start + (self.eps_end - self.eps_start) * (global_step / duration)

            a = self.select_action(s, epsilon=epsilon)
            s_next, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            self.save_to_buffer(s, a, r, s_next, terminated, truncated)

            if global_step % train_schedule == 0:
                self.train(s, a, r, s_next, terminated)

            if global_step % eval_schedule == 0:
                s, _ = env.reset()
                done, eval_return = False, 0.

                while not done:
                    # set epsilon = 0 to make an agent act greedy
                    a = self.select_action(s, epsilon=0.)
                    s_next, r, terminated, truncated, _ = env.step(a)
                    done = terminated or truncated
                    eval_return += r
                    s = s_next

                    if done:
                        break

                eval_return_history.append(eval_return)
                avg_return = np.mean(eval_return_history)
                print(f'{global_step=} | {avg_return=:.3f} | {epsilon=:.3f}')

            s = s_next
            if done:
                s, _ = env.reset()
                done = False

        return self
    
    def get_epsilon(self, eps_dur, total_steps):
        """Compute epsilon for current step using linear decay."""
        if self.global_step >= eps_dur * total_steps:
            return self.eps_end
        return self.eps_start + (self.eps_end - self.eps_start) * (self.global_step / (eps_dur * total_steps))


class PrioritizedDQNAgent(DQNAgent):
    """DQN agent with prioritized experience replay."""
    
    def __init__(self, *args, **kwargs):
        """Initialize prioritized DQN agent."""
        super().__init__(*args, **kwargs)
        # Change replay buffer structure to include priorities
        self.replay_buffer = deque(maxlen=kwargs.get('replay_buffer_size', 10000))
    
    def store_transition(self, state, action, reward, next_state, done, priority=1.0):
        """Store transition with priority in replay buffer."""
        self.replay_buffer.append((priority, state, action, reward, next_state, done))
    
    def symlog(self, x):
        """Apply symlog transformation to priorities."""
        return np.sign(x) * np.log(np.abs(x) + 1)
    
    def softmax(self, xs, temp=1.):
        """Compute softmax probabilities."""
        exp_xs = np.exp((xs - xs.max()) / temp)
        return exp_xs / exp_xs.sum()
    
    def sample_prioritized_batch(self, batch_size=None):
        """Sample batch with prioritization based on TD error."""
        if batch_size is None:
            batch_size = self.batch_size
        
        if len(self.replay_buffer) < batch_size:
            batch_size = len(self.replay_buffer)
        
        # Extract priorities
        priorities = np.array([rep[0] for rep in self.replay_buffer])
        probas = self.softmax(self.symlog(priorities))
        
        # Sample indices based on priorities
        indices = np.random.choice(len(self.replay_buffer), size=batch_size, 
                                   replace=False, p=probas)
        
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for idx in indices:
            _, state, action, reward, next_state, done = self.replay_buffer[idx]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones)), indices

    def save_to_buffer(self, state, action, reward, next_state, terminated, truncated):
        """Store transition in replay buffer with initial priority."""
        td_target = compute_td_target(self.Q, [reward], [next_state], [terminated], gamma=self.gamma)
        _, td_losses = compute_td_loss(self.Q, [state], [action], td_target)
        loss = td_losses.item()

        self.replay_buffer.append((loss, state, action, reward, next_state, terminated))
    
    def train(self, s, a, r, s_next, terminated):
        """Perform one training step with prioritized replay."""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        (states, actions, rewards, next_states, dones), indices = self.sample_prioritized_batch()
        
        self.optimizer.zero_grad()
        td_target = compute_td_target(self.Q, rewards, next_states, dones, gamma=self.gamma)
        loss, td_losses = compute_td_loss(self.Q, states, actions, td_target)
        loss.backward()
        self.optimizer.step()
        
        # Update priorities
        for i, idx in enumerate(indices):
            priority, state, action, reward, next_state, done = self.replay_buffer[idx]
            new_priority = td_losses[i].item() if isinstance(td_losses[i], torch.Tensor) else td_losses[i]
            self.replay_buffer[idx] = (new_priority, state, action, reward, next_state, done)
        
        return loss.item()
    