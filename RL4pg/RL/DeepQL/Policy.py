import numpy as np
import torch
import random
import torch.nn.functional as F
import math

class Policy:
    """ 
    Base Q Learning Policy class

    """
    def __init__(self, action_space_dim):
        self.action_space_dim=action_space_dim

    def select_action(self, q_values):
        """
        Select the action based on the q_values

        - q_values (torch.tensor)
        """
        pass

    def update(self):
        """
        manage the decay of the exploration parameters
        """
        pass



class EpsilonGreedyPolicy(Policy):
    """
    Epsilon-Greedy Policy.
    
    Attributes:
        epsilon (float): Probability of selecting a random action.
    """
    def __init__(self, action_space_dim, epsilon=1 ,  min_epsilon= 0.1 , decay_mode="half-life", half_life=200 , decay=0.99  ):
        super().__init__(action_space_dim)
        self.epsilon = epsilon
        self.epsilon_init=epsilon
        self.decay=decay
        self.min_epsilon=min_epsilon
        self.half_life=half_life/math.log(2)  # so when the timesteps are equal to the halflife then exp(-t*ln(2)/t)=exp(-ln(2))=1/2
        self.decay_mode=decay_mode
        self.time_step=0

    def select_action(self, q_values):
        """
        Selects an action using the ε-greedy strategy.

        Args:
            q_values (np.ndarray): Array of Q-values.

        Returns:
            int: Selected action.
        """
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.action_space_dim- 1), True # Random action
        else:
            return torch.argmax(q_values).item(), False  # Greedy action
        
    def _decay_halflife(self):
        self.time_step+=1
        self.epsilon=self.min_epsilon+ (self.epsilon_init-self.min_epsilon)* np.exp(-self.time_step/self.half_life)

    def _decay_multiplicative(self):
        self.epsilon=max(self.min_epsilon, self.epsilon * self.decay)

        
    def update(self):
        if self.decay_mode=="half-life":
            return self._decay_halflife()
        else:
            return self._decay_multiplicative()



class BoltzmannPolicy(Policy):
    """
    Boltzmann Exploration Policy (Softmax Sampling).

    Attributes:
        temperature (float): Controls exploration vs exploitation.
    """
    def __init__(self, action_space_dim, temperature=1.0, decay=0.99, min_temperature=0.1):
        super().__init__(action_space_dim)
        self.temperature = temperature
        self.decay=decay
        self.min_temperature=min_temperature

    def select_action(self, q_values):
        """
        Selects an action using the Boltzmann (softmax) strategy.

        Args:
            q_values (torch.Tensor): Tensor of Q-values.

        Returns:
            int: Selected action.
        """
        probs = F.softmax(q_values / self.temperature, dim=0)  # Compute softmax probabilities
        return torch.multinomial(probs, 1).item()  # Sample an action
    

    def update(self):
        """
        Decays temperature after each episode or step.

        Ensures temperature does not fall below `min_temperature`.
        """
        self.temperature = max(self.min_temperature, self.temperature * self.decay)