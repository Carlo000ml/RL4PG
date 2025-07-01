from torch_geometric.data import Batch
from collections import deque
import random
import torch
import numpy as np
from ..utils import SumTree


class Buffer:
    def __init__(self,capacity, device="cpu"):
        self.capacity=capacity
        self.device=device

    def add(self):
        pass
    def sample(self,batch_size):
        pass


class BasicReplayBuffer(Buffer):
    def __init__(self, capacity, device='cpu'):
        super().__init__(capacity=capacity,device=device)
        self.buffer = deque(maxlen=capacity)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self,batch_size, way="g-a-r-ng-d", demonstrations=False):
        batch = random.sample(self.buffer, batch_size)
        batch=process_batch(batch=batch,way=way,demonstrations=demonstrations,device=self.device)
        return batch
    def __len__(self):
        return len(self.buffer)

    

    





class Basic_PrioritizedReplayBuffer(Buffer):
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=1e-4, abs_err_upper=1.0, device="cpu"):
        """
        capacity: maximum number of experiences
        alpha: how much prioritization is used (0 means no prioritization)
        beta: importance-sampling exponent (to correct the bias)
        beta_increment: incremental amount to anneal beta towards 1
        abs_err_upper: maximum absolute error (for clipping)
        """
        super().__init__(capacity=capacity,device=device)
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.abs_err_upper = abs_err_upper
        self.epsilon = 1e-6  # Small constant to ensure non-zero priority

    def _get_priority(self, error):
        """Compute priority value given TD error."""
        return (np.abs(error) + self.epsilon) ** self.alpha

    def add(self, experience): 
        """
        experience: tuple (state, action, reward, next_state, done)
        error: TD error for the experience (not used here; new experiences use the max priority)
        When storing a new experience, use the maximum priority observed so far.
        If the buffer is empty, default to 1.
        """
        
        if self.tree.size == 0:
            max_priority = 1.0
        else:
            # The leaves are stored from index (capacity - 1) to (capacity - 1 + size)
            leaf_start = self.tree.capacity - 1
            max_priority = np.max(self.tree.tree[leaf_start:leaf_start + self.tree.size])
        
        self.tree.add(max_priority, experience)


    def sample(self, n, way="g-a-r-ng-d", demonstrations=False):
        """
        Sample a batch of experiences.
        Returns: indices, batch (list of experiences), and importance-sampling (IS) weights.
        """
        batch = []
        idxs = []
        priorities = []
        segment = self.tree.total_priority / n

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, priority, data = self.tree.get_leaf(s)
            batch.append(data)
            priorities.append(priority)
            idxs.append(idx)

        sampling_probabilities = np.array(priorities) / self.tree.total_priority

        # Anneal beta towards 1
        self.beta = min(1.0, self.beta + self.beta_increment)

        # Compute importance-sampling weights to correct the bias
        weights = (self.tree.size * sampling_probabilities) ** (-self.beta)
        weights /= weights.max()  # Normalize for stability

        batch=process_batch(batch=batch,way=way,demonstrations=demonstrations,device=self.device)

        return idxs, batch, weights
    

    def update(self, idxs, errors):
        """
        Update the priorities of sampled experiences.
        idxs: list of indices for the experiences in the sum tree
        errors: new TD errors for the sampled experiences
        """
        for idx, error in zip(idxs, errors):
            priority = self._get_priority(error)
            # Optionally clip the priority to avoid excessively high values
            priority = min(priority, self.abs_err_upper)
            self.tree.update(idx, priority)

    def __len__(self):
        return self.tree.size




def process_batch(batch, way="s-a-r-ns-d", demonstrations=False,  device="cpu"):
    if demonstrations:
        if way=="g-a-r-ng-d":
            states, actions, rewards, done_t,state_t_1, state_t_n, done_n=zip(*batch)

            batched_states = Batch.from_data_list(states)
            batched_states_t_1 = Batch.from_data_list(state_t_1)
            batched_states_t_n = Batch.from_data_list(state_t_n)

            return (batched_states, 
                    torch.tensor(actions, dtype=torch.long, device=device).view(-1,1),
                    torch.vstack(rewards).to(device),
                    batched_states_t_1, 
                    torch.tensor(done_t, dtype=torch.long, device=device),
                    batched_states_t_n,
                    torch.tensor(done_n, dtype=torch.long, device=device)
            )
            
        if way=="s-a-r-ns-d":
            list_of_states, actions, rewards,state_t_1, done_t, list_of_states_tn, done_n=zip(*batch)
            return (
                torch.stack(list_of_states).to(device),
                torch.tensor(actions, dtype=torch.long,device=device).view(-1,1),
                torch.vstack(rewards).to(device),
                torch.stack(state_t_1).to(device), 
                torch.tensor(done_t, dtype=torch.long, device=device),
                torch.stack(list_of_states_tn).to(device),
                torch.tensor(done_n, dtype=torch.long, device=device)
            )
            
        

    else:   # no demonstrations and no rew decomposer
        if way=="g-a-r-ng-d":  # g for graph
            states, actions, rewards, next_states, dones = zip(*batch)
            # Batch the graph data using torch_geometric's Batch
            batched_states = Batch.from_data_list(states)
            batched_next_states = Batch.from_data_list(next_states)
            return (
                batched_states,  
                torch.tensor(actions, dtype=torch.long, device=device).view(-1,1),
                torch.tensor(rewards, dtype=torch.float, device=device), 
                batched_next_states,
                torch.tensor(dones, dtype=torch.long, device=device),
                0,
                0
            )
            
        if way=="s-a-r-ns-d":  # s for state, a torch tensor -> for the manager
            list_of_states, list_of_actions , list_of_rewards , list_of_next_states, list_of_dones= zip(*batch)
            return (
                torch.stack(list_of_states).to(device) , 
                torch.tensor(list_of_actions, dtype=torch.long, device=device).view(-1, 1),
                torch.tensor(list_of_rewards, dtype=torch.float32, device=device), 
                torch.stack(list_of_next_states).to(device),
                torch.tensor(list_of_dones, dtype=torch.long, device=device),
                0, 
                0
            )
            
    print("Input Error")
    return None




################################ OLD IMPLEMENTATION

class GraphReplayBuffer(Buffer):
    def __init__(self, capacity, device='cpu'):
        super().__init__(capacity=capacity,device=device)
        self.buffer = deque(maxlen=capacity)

    def add_2_step(self,state, action, reward_t, reward_t_1, state_t_2, done_t_2):
        self.buffer.append((state, action, reward_t, reward_t_1, state_t_2, done_t_2))

    def add(self, state, action, reward, next_state, done):
        """
        Add a transition to the buffer.
        Args:
            state (torch_geometric.data.Data): The current graph state. already put on the right device.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (torch_geometric.data.Data): The next graph state. already put on the rigth device.
            done (bool): Whether the episode is done.
        """
        self.buffer.append((state, action, reward, next_state, done))


    def add_demonstration(self, demonstration):
        self.buffer.append(demonstration)


    def add_multiple_experiences(self,list_of_exp:list):
        self.buffer.extend(list_of_exp)

    def sample(self, batch_size):
        """
        Sample a batch of transitions and batch the graph states. random.sample() avoid replacement by default
        Returns:
            Batch of graph states, actions, rewards, next_states, and dones.
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Batch the graph data using torch_geometric's Batch
        batched_states = Batch.from_data_list(states)
        batched_next_states = Batch.from_data_list(next_states)
        if batched_states.x.device.type=="cpu":
            batched_states.x=batched_states.x.to(self.device)
            batched_states.edge_index=batched_states.edge_index.to(self.device)
            batched_next_states.x=batched_next_states.x.to(self.device)
            batched_next_states.edge_index=batched_next_states.edge_index.to(self.device)
       

        return (
            batched_states,
            torch.tensor(actions, dtype=torch.long, device=self.device),
            torch.tensor(rewards, dtype=torch.float, device=self.device),
            batched_next_states,
            torch.tensor(dones, dtype=torch.long, device=self.device),
        )
    
    def sample_demonstrations(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, done_t, state_t_1, state_t_n, done_n=zip(*batch)

        batched_states = Batch.from_data_list(states)
        batched_states_t_1 = Batch.from_data_list(state_t_1)
        batched_states_t_n = Batch.from_data_list(state_t_n)

        return (
            batched_states,
            torch.tensor(actions, dtype=torch.long, device=self.device),
            torch.vstack(rewards).to(self.device),
            torch.tensor(done_t, dtype=torch.long, device=self.device),
            batched_states_t_1,
            batched_states_t_n,
            torch.tensor(done_n, dtype=torch.long, device=self.device),
        )
    
    def sample_2_step(self, batch_size):
        """
        Sample a batch of transitions and batch the graph states. random.sample() avoid replacement by default
        Returns:
            Batch of graph states, actions, rewards, next_states, and dones.
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, reward_ts, reward_t_1s, state_t_2s, done_t_2s = zip(*batch)

        # Batch the graph data using torch_geometric's Batch
        batched_states = Batch.from_data_list(states)
        batched_states_t_2 = Batch.from_data_list(state_t_2s)

        if batched_states.x.device.type=="cpu":
            batched_states.x=batched_states.x.to(self.device)
            batched_states.edge_index=batched_states.edge_index.to(self.device)
            batched_states_t_2.x=batched_states_t_2.x.to(self.device)
            batched_states_t_2.edge_index=batched_states_t_2.edge_index.to(self.device)
       

        return (
            batched_states,
            torch.tensor(actions, dtype=torch.long, device=self.device),
            torch.tensor(reward_ts, dtype=torch.float, device=self.device),
            torch.tensor(reward_t_1s, dtype=torch.float, device=self.device),
            batched_states_t_2,
            torch.tensor(done_t_2s, dtype=torch.long, device=self.device),
        )

    def __len__(self):
        return len(self.buffer)
    




class PrioritizedReplayBuffer(Buffer):
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=1e-4, abs_err_upper=1.0, device="cpu"):
        """
        capacity: maximum number of experiences
        alpha: how much prioritization is used (0 means no prioritization)
        beta: importance-sampling exponent (to correct the bias)
        beta_increment: incremental amount to anneal beta towards 1
        abs_err_upper: maximum absolute error (for clipping)
        """
        super().__init__(capacity=capacity,device=device)
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.abs_err_upper = abs_err_upper
        self.epsilon = 1e-6  # Small constant to ensure non-zero priority

    def _get_priority(self, error):
        """Compute priority value given TD error."""
        return (np.abs(error) + self.epsilon) ** self.alpha

    def add(self, graph, action, reward, next_graph, done ): #experience):
        """
        experience: tuple (state, action, reward, next_state, done)
        error: TD error for the experience (not used here; new experiences use the max priority)
        When storing a new experience, use the maximum priority observed so far.
        If the buffer is empty, default to 1.
        """
        
        if self.tree.size == 0:
            max_priority = 1.0
        else:
            # The leaves are stored from index (capacity - 1) to (capacity - 1 + size)
            leaf_start = self.tree.capacity - 1
            max_priority = np.max(self.tree.tree[leaf_start:leaf_start + self.tree.size])
        
        self.tree.add(max_priority, (graph, action, reward, next_graph, done))

    def add_2_step(self, state, action, reward_t, reward_t_1, state_t_2, done_t_2): #experience):
        """
        experience: tuple (state, action, reward, next_state, done)
        error: TD error for the experience (not used here; new experiences use the max priority)
        When storing a new experience, use the maximum priority observed so far.
        If the buffer is empty, default to 1.
        """
        
        if self.tree.size == 0:
            max_priority = 1.0
        else:
            # The leaves are stored from index (capacity - 1) to (capacity - 1 + size)
            leaf_start = self.tree.capacity - 1
            max_priority = np.max(self.tree.tree[leaf_start:leaf_start + self.tree.size])
        
        self.tree.add(max_priority, (state, action, reward_t, reward_t_1, state_t_2, done_t_2))

    def add_demonstration(self, demonstration):

        if self.tree.size == 0:
            max_priority = 1.0
        else:
            # The leaves are stored from index (capacity - 1) to (capacity - 1 + size)
            leaf_start = self.tree.capacity - 1
            max_priority = np.max(self.tree.tree[leaf_start:leaf_start + self.tree.size])
        
        self.tree.add(max_priority, demonstration)



    def sample(self, n):
        """
        Sample a batch of experiences.
        Returns: indices, batch (list of experiences), and importance-sampling (IS) weights.
        """
        batch = []
        idxs = []
        priorities = []
        segment = self.tree.total_priority / n

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, priority, data = self.tree.get_leaf(s)
            batch.append(data)
            priorities.append(priority)
            idxs.append(idx)

        sampling_probabilities = np.array(priorities) / self.tree.total_priority

        # Anneal beta towards 1
        self.beta = min(1.0, self.beta + self.beta_increment)

        # Compute importance-sampling weights to correct the bias
        weights = (self.tree.size * sampling_probabilities) ** (-self.beta)
        weights /= weights.max()  # Normalize for stability

        states, actions, rewards, next_states, dones = zip(*batch)

        # Batch the graph data using torch_geometric's Batch
        batched_states = Batch.from_data_list(states)
        batched_next_states = Batch.from_data_list(next_states)

        batch=(
            batched_states,
            torch.tensor(actions, dtype=torch.long, device=self.device),
            torch.tensor(rewards, dtype=torch.float, device=self.device),
            batched_next_states,
            torch.tensor(dones, dtype=torch.long, device=self.device),
        )

        return idxs, batch, weights
    

    def sample_2_step(self, n):
        """
        Sample a batch of experiences.
        Returns: indices, batch (list of experiences), and importance-sampling (IS) weights.
        """
        batch = []
        idxs = []
        priorities = []
        segment = self.tree.total_priority / n

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, priority, data = self.tree.get_leaf(s)
            batch.append(data)
            priorities.append(priority)
            idxs.append(idx)

        sampling_probabilities = np.array(priorities) / self.tree.total_priority

        # Anneal beta towards 1
        self.beta = min(1.0, self.beta + self.beta_increment)

        # Compute importance-sampling weights to correct the bias
        weights = (self.tree.size * sampling_probabilities) ** (-self.beta)
        weights /= weights.max()  # Normalize for stability

        states, actions, reward_ts, reward_t_1s, state_t_2s, done_t_2s= zip(*batch)

        # Batch the graph data using torch_geometric's Batch
        batched_states = Batch.from_data_list(states)
        batched_next_states = Batch.from_data_list(state_t_2s)

        batch=(
            batched_states,
            torch.tensor(actions, dtype=torch.long, device=self.device),
            torch.tensor(reward_ts, dtype=torch.float, device=self.device),
            torch.tensor(reward_t_1s, dtype=torch.float, device=self.device),
            batched_next_states,
            torch.tensor(done_t_2s, dtype=torch.long, device=self.device),
        )

        return idxs, batch, weights
    
    def sample_demonstrations(self, n):
        """
        Sample a batch of experiences.
        Returns: indices, batch (list of experiences), and importance-sampling (IS) weights.
        """
        batch = []
        idxs = []
        priorities = []
        segment = self.tree.total_priority / n

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, priority, data = self.tree.get_leaf(s)
            batch.append(data)
            priorities.append(priority)
            idxs.append(idx)

        sampling_probabilities = np.array(priorities) / self.tree.total_priority

        # Anneal beta towards 1
        self.beta = min(1.0, self.beta + self.beta_increment)

        # Compute importance-sampling weights to correct the bias
        weights = (self.tree.size * sampling_probabilities) ** (-self.beta)
        weights /= weights.max()  # Normalize for stability

        states, actions, rewards, done_t, state_t_1, state_t_n, done_n=zip(*batch)

        batched_states = Batch.from_data_list(states)
        batched_states_t_1 = Batch.from_data_list(state_t_1)
        batched_states_t_n = Batch.from_data_list(state_t_n)

        batch=(
            batched_states,
            torch.tensor(actions, dtype=torch.long, device=self.device),
            torch.vstack(rewards).to(self.device),
            torch.tensor(done_t, dtype=torch.long, device=self.device),
            batched_states_t_1,
            batched_states_t_n,
            torch.tensor(done_n, dtype=torch.long, device=self.device),
        )
    

        return idxs, batch, weights
    

    def update(self, idxs, errors):
        """
        Update the priorities of sampled experiences.
        idxs: list of indices for the experiences in the sum tree
        errors: new TD errors for the sampled experiences
        """
        for idx, error in zip(idxs, errors):
            priority = self._get_priority(error)
            # Optionally clip the priority to avoid excessively high values
            priority = min(priority, self.abs_err_upper)
            self.tree.update(idx, priority)

    def __len__(self):
        return self.tree.size