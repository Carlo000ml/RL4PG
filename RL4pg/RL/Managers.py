import torch
import numpy as np
import random
from .DeepQL.Q_estimators import DQNetwork, DuelingQNetwork
from ..RL.DeepQL.Policy import BoltzmannPolicy, EpsilonGreedyPolicy
from ..RL.ReplyBuffers import Basic_PrioritizedReplayBuffer, BasicReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from ..utils import N_1_secure_action_space





class MultiAgent_Line_Manager:
    """
    A manager class to control multiple agents in a reinforcement learning environment for power grid management.

    This class provides functionality for determining the safety of the grid, selecting agents 
    for actions, and managing reconnections of disconnected lines.

    Attributes:
        env (grid2op.Environment.Environment): 
            The grid2op environment in which the agents operate.
        rho_threshold (float): 
            The threshold for line load (rho) above which the grid is considered unsafe. Default is 0.5.
        connection_flag (bool): 
            If true, whenever a line is disconnected the action performed is the reconnection of it.

    Methods:
        safe(obs, threshold):
            Determines whether the grid is in a safe state based on the threshold, if the threshold is not provided it uses the initialization threshold.
        select_candidate_agent(obs):
            Select the most risky line
        check_disconnections(obs):
            Checks if there are any disconnected lines in the grid.
        reconnect_line(obs):
            Reconnects a randomly selected disconnected line.
        do_nothing():
            Returns a "do-nothing" action.

    """
    def __init__(self,environment, rho_threshold=0.7 , connect_disconnected_line=True):
        """
        Initializes the MultiAgent_Manager.

        Args:
            environment (grid2op.Environment.Environment): 
                The grid2op environment to manage.
            rho_threshold (float, optional): 
                The threshold for line loading (rho) to determine grid safety. Default is 0.5.
            connect_disconnected_line (bool, optional): 
                If true, whenever a line is disconnected the action performed is the reconnection of it.
        """
        self.act_space=environment.action_space
        self.rho_threshold=rho_threshold
        self.connection_flag=connect_disconnected_line


    def safe(self,obs, threshold=None):
        """
        Checks if the grid is in a safe state based on the rho threshold.

        Args:
            obs (grid2op.Observation.Observation): 
                The current observation of the grid.

        Returns:
            bool: True if all lines are within the rho threshold, False otherwise.
        """
        if threshold is None:
            return np.all(obs.rho<=self.rho_threshold)
        else:
            return np.all(obs.rho<=threshold)

    
    def select_candidate_agent(self,obs):
        """
        Identifies two candidate substations (origin and extremity) based on the most overloaded line.

        Args:
            obs (grid2op.Observation.Observation): 
                The current observation of the grid.

        Returns:
            tuple: A tuple containing:
                - `sub_or` (int): The substation ID at the origin of the overloaded line.
                - `sub_ex` (int): The substation ID at the extremity of the overloaded line.
        """
        line_id=np.argmax(obs.rho)
        
        return line_id
    
    
    def check_disconnections(self, obs):
        """
        Checks if there is any disconnected line in the grid.

        Args:
            obs (grid2op.Observation.Observation): 
                The current observation of the grid.

        Returns:
            bool: True if there is at least one disconnected line, False otherwise.
        """
        return np.any(obs.rho==0.0)
    
     
    
    def reconnect_line(self,obs):
        """
        Reconnects a random disconnected line.

        Args:
            obs (grid2op.Observation.Observation): 
                The current observation of the grid.

        Returns:
            grid2op.Action.Action: The action to reconnect a randomly selected disconnected line.

        Raises:
            AssertionError: If there are no disconnected lines to reconnect.
        """
        line=np.where(obs.rho==0.0)[0]
        assert len(line)>0
        line_id=random.choice(line)
        return self.act_space.reconnect_powerline(line_id=line_id, bus_or=0,bus_ex=0)

    
    def do_nothing(self):
        """
        Returns a "do-nothing" action.

        Returns:
            grid2op.Action.Action: The "do-nothing" action.
        """
        return self.act_space({})
    








class MultiAgent_RL_Line_Manager:
    """

    """
    def __init__(self,
                    environment,
                    runs_name,

                    Q_estimator_net_type, 
                    Q_estimator_params, 
                    lr, 
                    weight_decay, 
                    gradient_clipping,
                    tau_soft_updates,
                    soft_updates,
                    use_double,
                    gamma=0.999,

                    policy_type="epsilon-greedy",
                    policy_kargs={},

                    replay_buffer_kargs={},
                    buffer_type="basic",

                    start_training_capacity=2000,
                    num_training_iters=150,
                    batch_size=32,

                    rho_threshold=0.75 , 
                    connect_disconnected_line=True, 
                    log_loss=True,

                    learn_from_demonstrations=False,
                    min_demonstrations=500,
                    margin=1,
                    lambda1=0.5,
                    lambda2=0.1,
                    lambda3=0.1,

                    device="cpu", 
                    **kwargs):
            """
            Initializes the MultiAgent_Manager.

            Args:
                environment (grid2op.Environment.Environment): 
                    The grid2op environment to manage.
                rho_threshold (float, optional): 
                    The threshold for line loading (rho) to determine grid safety.
                connect_disconnected_line (bool, optional): 
                    If true, whenever a line is disconnected the action performed is the reconnection of it.
            """
            assert policy_type in ["epsilon-greedy" , "boltzmann"]

            self.act_space=environment.action_space
            self.rho_threshold=rho_threshold
            self.connection_flag=connect_disconnected_line


            self.action_space_dim=environment.n_line

            self.device=device

            Q_estimator_params["output_dim"]=self.action_space_dim

            if Q_estimator_net_type=="Dueling":
                self.main_net=DuelingQNetwork(**Q_estimator_params).to(device)
                self.target_net = DuelingQNetwork(**Q_estimator_params).to(device)
                self.target_net.load_state_dict(self.main_net.state_dict())
                self.target_net.eval()
            else:
                self.main_net=DQNetwork(**Q_estimator_params).to(device)
                self.target_net = DQNetwork(**Q_estimator_params).to(device)
                self.target_net.load_state_dict(self.main_net.state_dict())
                self.target_net.eval()

            # optimizer
            self.optimizer = torch.optim.Adam(self.main_net.parameters(), lr=lr, weight_decay=weight_decay)
            self.gradient_clipping=gradient_clipping
            self.tau_soft_updates=tau_soft_updates
            self.soft_updates=soft_updates
            self.use_double=use_double
            self.gamma=gamma
            self.loss = torch.nn.MSELoss()
            self.log_loss=log_loss


            # policy 
            self.policy_type=policy_type
            if policy_type=="epsilon-greedy": 
                policy_kargs["action_space_dim"]=self.action_space_dim
                self.policy=EpsilonGreedyPolicy(**policy_kargs)
            else: 
                policy_kargs["action_space_dim"]=self.action_space_dim
                self.policy=BoltzmannPolicy(**policy_kargs)


            # replay buffer
            assert buffer_type in [ "basic" , "prioritized"]
            self.buffer_type=buffer_type
            replay_buffer_kargs["device"]=self.device
            if self.buffer_type=="basic":
                self.buffer=BasicReplayBuffer(**replay_buffer_kargs)
                self.demonstration_buffer=BasicReplayBuffer(**replay_buffer_kargs)
                assert learn_from_demonstrations==False
            elif self.buffer_type=="prioritized":
                self.buffer=Basic_PrioritizedReplayBuffer(**replay_buffer_kargs)
                self.demonstration_buffer=BasicReplayBuffer(**replay_buffer_kargs)

            self.learn_from_demonstrations=learn_from_demonstrations
            self.min_demonstrations=min_demonstrations
            self.margin=margin
            self.lambda1=lambda1
            self.lambda2=lambda2
            self.lambda3=lambda3


            self.start_training_capacity=start_training_capacity
            self.num_training_iters=num_training_iters
            self.batch_size=batch_size


            # counters

            self.training_iter=0
            self.training_count=0
            self.effective_training_iter=0
            self.demonstration_training_iter=0

            # writer
            self.ma_path=runs_name+"/"+"Manager"
            self.writer = SummaryWriter(log_dir=self.ma_path)




    def safe(self,obs, threshold=None):
        """
        Checks if the grid is in a safe state based on the rho threshold.

        Args:
            obs (grid2op.Observation.Observation): 
                The current observation of the grid.

        Returns:
            bool: True if all lines are within the rho threshold, False otherwise.
        """
        if threshold is None:
            return np.all(obs.rho<=self.rho_threshold)
        else:
            return np.all(obs.rho<=threshold)
        
    def compute_main_q_values(self , input, train=True):
        if train:
            self.main_net.train()
        else:
            self.main_net.eval()

        return self.main_net(input)
    
    def compute_target_q_values(self , input):

        self.target_net.eval()
        with torch.no_grad():
            out=self.target_net(input).detach()

        return out

        
    def exploit_agent(self,obs):
        obs=self.obs_to_torch(obs).unsqueeze(0)  # torch vector
        q_values=self.compute_main_q_values(obs, train=False)
        return torch.argmax(q_values).item()
    
    def play_agent(self,obs):
        obs=self.obs_to_torch(obs).unsqueeze(0)  # torch vector
        q_values=self.compute_main_q_values(obs, train=False)
        return self.policy.select_action(q_values)
    
    def store_experience(self,experience):
        self.buffer.add(experience=experience)

    def store_demonstration(self,demonstration): # g-a-r-ng-d-nng-nd 
        """
        just to remind how the sampling works: s-a-r-ns-d-nng-nd
        """
        self.demonstration_buffer.add(demonstration)
       
        self.buffer.add((demonstration[0] ,demonstration[1] , demonstration[2][0], demonstration[3] ,demonstration[4]))


    def _training_iteration(self):
        if self.buffer_type=="basic":
            states, actions , rewards , next_states, dones , _ , _ =self.buffer.sample(self.batch_size,  way="s-a-r-ns-d", demonstrations=False)
            loss,_=self.compute_loss(batch_state=states,batch_action=actions,batch_reward=rewards,batch_next_state=next_states,batch_done=dones)

        else:
            idxs, batch, weights=self.buffer.sample(self.batch_size,way="s-a-r-ns-d", demonstrations=False)
            states, actions , rewards , next_states, dones , _ , _= batch

            loss, td_error=self.compute_loss(batch_state=states,batch_action=actions,batch_reward=rewards,batch_next_state=next_states,batch_done=dones, IS_weights=weights)
            self.buffer.update(idxs=idxs, errors=td_error)

        self.writer.add_scalar("Loss",loss.item() , self.training_iter)
        self.update_parameters(loss=loss)


    def _demonstration_training_iteration(self):
        # sample a batch
        states, actions , rewards , next_states, dones ,states_tn, dones_tn = self.demonstration_buffer.sample(self.batch_size, way="s-a-r-ns-d", demonstrations=True)
        loss,_=self.compute_demonstration_loss(states=states,actions=actions,rewards=rewards,next_states=next_states,dones=dones, states_tn=states_tn , dones_tn=dones_tn)


        if self.log_loss: self.writer.add_scalar("Demonstration Loss",loss.item() , self.demonstration_training_iter)

        # update parameters
        self.update_parameters(loss)
        self.demonstration_training_iter+=1
    


    def update_parameters(self,loss):
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.main_net.parameters(), self.gradient_clipping)
        self.optimizer.step()



    def compute_loss(self, batch_state, batch_action, batch_reward, batch_next_state, batch_done, IS_weights=None):
        ### note the state must already have been processed by the graph
        ### this function returns only the loss, another function for the optimizer
        if IS_weights is None and self.buffer_type=="basic":

            actions=batch_action
            q_values=self.compute_main_q_values(batch_state, train=True)
            q_values=torch.gather(q_values, 1, actions).view(-1)

            with torch.no_grad():
                if self.use_double:
                    best_main_actions=self.compute_main_q_values(batch_next_state, train=False).argmax(dim=1, keepdim=True).detach()
                    target_q_values=self.compute_target_q_values(batch_next_state).gather(1, best_main_actions).view(-1)
                else:
                    target_q_values=self.compute_target_q_values(batch_next_state)
                    target_q_values, _ = torch.max(target_q_values, dim=1)

                target_q_values= batch_reward+ (1-batch_done) * self.gamma * target_q_values

            return self.loss(q_values,target_q_values), None
        else:
            actions=batch_action
            q_values=self.compute_main_q_values(batch_state, train=True)
            q_values=torch.gather(q_values, 1, actions).view(-1)

            with torch.no_grad():
                if self.use_double:
                    best_main_actions=self.compute_main_q_values(batch_next_state, train=False).argmax(dim=1, keepdim=True).detach()
                    target_q_values=self.compute_target_q_values(batch_next_state).gather(1, best_main_actions).view(-1)
                else:
                    target_q_values=self.compute_target_q_values(batch_next_state)
                    target_q_values, _ = torch.max(target_q_values, dim=1)

                target_q_values= batch_reward+ (1-batch_done) * self.gamma * target_q_values

            TD_errors=target_q_values-q_values
            loss = (torch.tensor(IS_weights, device=self.device) * TD_errors.pow(2)).sum()


            return loss , TD_errors.detach().cpu().numpy()
        

    def compute_demonstration_loss(self, states, actions, rewards, next_states, dones, states_tn, dones_tn):
        ####### standard DQN loss   Note: only prioritized buffer


        q_values=self.compute_main_q_values(states, train=True)
        batch_size,n_actions=q_values.shape  # number of rows , number of columns of the q_values
        q_values=torch.gather(q_values, 1, actions).view(-1)
        n=rewards.shape[1]   # number of time step considered

        with torch.no_grad():
                if self.use_double:
                    best_main_actions=self.compute_main_q_values(next_states, train=False).argmax(dim=1, keepdim=True).detach()
                    target_q_values=self.compute_target_q_values(next_states).gather(1, best_main_actions).view(-1)
                else:
                    target_q_values=self.compute_target_q_values(next_states)
                    target_q_values, _ = torch.max(target_q_values, dim=1)

                target_q_values= (1-dones) * self.gamma * target_q_values
        target_q_values= target_q_values + rewards[:,0]

        single_step_loss=self.loss(q_values,target_q_values)

        with torch.no_grad():
            if self.use_double:
                best_main_actions=self.compute_main_q_values(states_tn, train=False).argmax(dim=1, keepdim=True).detach()
                target_q_values=self.compute_target_q_values(states_tn).gather(1, best_main_actions).view(-1)
            else:
                target_q_values=self.compute_target_q_values(states_tn)
                target_q_values, _ = torch.max(target_q_values, dim=1)

            #computation of the n step reward
            exponents=torch.arange(n).float()
            rewards=torch.sum(rewards* self.gamma**exponents, axis=1)  # shape (batch_size)

            target_q_values= rewards+ self.gamma**n  * target_q_values * (1-dones_tn)

        n_step_loss=self.loss(q_values,target_q_values)

        # supervised loss
        margin= torch.zeros(batch_size, n_actions) + self.margin
        indexes=torch.arange(batch_size)
        margin[indexes,actions.view(-1)]=0  #  place the zeros on the expert actions
        supervised_loss=torch.clip(torch.max(self.compute_main_q_values(states, train=True)+ margin ) - q_values,0).mean()

        # regularization loss
        l2_norm = sum(param.pow(2.0).sum() for param in self.main_net.parameters())


        return single_step_loss+self.lambda1*n_step_loss+self.lambda2* supervised_loss+self.lambda3* l2_norm, None


    def learn(self):

        if len(self.buffer)<self.start_training_capacity:
            self.training_iter+=self.num_training_iters
            self.training_count+=1
            return

        else:
            for i in range(self.num_training_iters):
                self._training_iteration()     
                self.training_iter+=1

            self.training_count+=1
            self.effective_training_iter+=self.num_training_iters


        if self.policy_type=="epsilon-greedy":
            self.writer.add_scalar("Epsilon",self.policy.epsilon , self.training_count)

        else:
            self.writer.add_scalar("Temperature",self.policy.temperature , self.training_count)

        self.policy.update()


    def learn_demonstrations(self):
        if len(self.demonstration_buffer)<self.min_demonstrations:
            return
        else:
            for i in range(self.num_training_iters):
                self._demonstration_training_iteration()
            


    def sync_target(self):
        if self.soft_updates:
            soft_update(self.target_net, self.main_net, self.tau_soft_updates)
        else:
            self.target_net.load_state_dict(self.main_net.state_dict())


    


    def save_checkpoint(self):
        torch.save({
        'main_net_state_dict': self.main_net.state_dict(),
        'target_net_state_dict': self.target_net.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict(),
        }, self.ma_path+"/checkpoint.pth")


    def load_checkpoint(self):
        checkpoint = torch.load(self.ma_path+"/checkpoint.pth")
        self.main_net.load_state_dict(checkpoint["main_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])



    
    def select_candidate_agent(self,obs):
        """

        Args:
            obs (grid2op.Observation.Observation): 
                The current observation of the grid.

        Returns:
            tuple: A tuple containing:
                - `sub_or` (int): The substation ID at the origin of the overloaded line.
                - `sub_ex` (int): The substation ID at the extremity of the overloaded line.
        """
        line_id=np.argmax(obs.rho)
        
        return line_id
    
    
    def check_disconnections(self, obs):
        """
        Checks if there is any disconnected line in the grid.

        Args:
            obs (grid2op.Observation.Observation): 
                The current observation of the grid.

        Returns:
            bool: True if there is at least one disconnected line, False otherwise.
        """
        return np.any(obs.rho==0.0)
    
     
    
    def reconnect_line(self,obs):
        """
        Reconnects a random disconnected line.

        Args:
            obs (grid2op.Observation.Observation): 
                The current observation of the grid.

        Returns:
            grid2op.Action.Action: The action to reconnect a randomly selected disconnected line.

        Raises:
            AssertionError: If there are no disconnected lines to reconnect.
        """
        line=np.where(obs.rho==0.0)[0]
        assert len(line)>0
        line_id=random.choice(line)
        return self.act_space.reconnect_powerline(line_id=line_id, bus_or=0,bus_ex=0)

    
    def do_nothing(self):
        """
        Returns a "do-nothing" action.

        Returns:
            grid2op.Action.Action: The "do-nothing" action.
        """
        return self.act_space({})
    
    def obs_to_torch(self,obs):
        return torch.tensor(np.concatenate([obs.rho, obs.a_or, obs.a_ex, obs.topo_vect],axis=0), device=self.device, dtype=torch.float32)




def playable_substations(env, n_1=True):
    playable=[]
    for i in range(env.n_sub):
        if n_1:
            acsp=N_1_secure_action_space(env,i)
        else:
            acsp=env.action_space.get_all_unitary_topologies_set(env.action_space, sub_id=i,add_alone_line=False)

        if len(acsp)!=0:
            playable.append(i)

    return playable



class MultiAgent_RL_Sub_Manager:
    """

    """

    def __init__(self,
                 environment,
                 runs_name,

                 Q_estimator_net_type,
                 Q_estimator_params,
                 lr,
                 weight_decay,
                 gradient_clipping,
                 tau_soft_updates,
                 soft_updates,
                 use_double,
                 gamma=0.999,

                 policy_type="epsilon-greedy",
                 policy_kargs={},

                 replay_buffer_kargs={},
                 buffer_type="basic",

                 start_training_capacity=2000,
                 num_training_iters=150,
                 batch_size=32,

                 rho_threshold=0.75,
                 connect_disconnected_line=True,
                 log_loss=True,

                 learn_from_demonstrations=False,
                 min_demonstrations=500,
                 margin=1,
                 lambda1=0.5,
                 lambda2=0.1,
                 lambda3=0.1,

                 device="cpu",
                 **kwargs):
        """
        Initializes the MultiAgent_Manager.

        Args:
            environment (grid2op.Environment.Environment):
                The grid2op environment to manage.
            rho_threshold (float, optional):
                The threshold for line loading (rho) to determine grid safety.
            connect_disconnected_line (bool, optional):
                If true, whenever a line is disconnected the action performed is the reconnection of it.
        """
        assert policy_type in ["epsilon-greedy", "boltzmann"]

        # action converters
        self.playable = playable_substations(environment)

        self.act_space = environment.action_space
        self.rho_threshold = rho_threshold
        self.connection_flag = connect_disconnected_line

        self.action_space_dim = len(self.playable)

        self.device = device

        Q_estimator_params["output_dim"] = self.action_space_dim

        if Q_estimator_net_type == "Dueling":
            self.main_net = DuelingQNetwork(**Q_estimator_params).to(device)
            self.target_net = DuelingQNetwork(**Q_estimator_params).to(device)
            self.target_net.load_state_dict(self.main_net.state_dict())
            self.target_net.eval()
        else:
            self.main_net = DQNetwork(**Q_estimator_params).to(device)
            self.target_net = DQNetwork(**Q_estimator_params).to(device)
            self.target_net.load_state_dict(self.main_net.state_dict())
            self.target_net.eval()

        # optimizer
        self.optimizer = torch.optim.Adam(self.main_net.parameters(), lr=lr, weight_decay=weight_decay)
        self.gradient_clipping = gradient_clipping
        self.tau_soft_updates = tau_soft_updates
        self.soft_updates = soft_updates
        self.use_double = use_double
        self.gamma = gamma
        self.loss = torch.nn.MSELoss()
        self.log_loss = log_loss

        # policy
        self.policy_type = policy_type
        if policy_type == "epsilon-greedy":
            policy_kargs["action_space_dim"] = self.action_space_dim
            self.policy = EpsilonGreedyPolicy(**policy_kargs)
        else:
            policy_kargs["action_space_dim"] = self.action_space_dim
            self.policy = BoltzmannPolicy(**policy_kargs)

        # replay buffer
        assert buffer_type in ["basic", "prioritized"]
        self.buffer_type = buffer_type
        replay_buffer_kargs["device"] = self.device
        if self.buffer_type == "basic":
            self.buffer = BasicReplayBuffer(**replay_buffer_kargs)
            self.demonstration_buffer = BasicReplayBuffer(**replay_buffer_kargs)
            assert learn_from_demonstrations == False
        elif self.buffer_type == "prioritized":
            self.buffer = Basic_PrioritizedReplayBuffer(**replay_buffer_kargs)
            self.demonstration_buffer = BasicReplayBuffer(**replay_buffer_kargs)

        self.learn_from_demonstrations = learn_from_demonstrations
        self.min_demonstrations = min_demonstrations
        self.margin = margin
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3

        self.start_training_capacity = start_training_capacity
        self.num_training_iters = num_training_iters
        self.batch_size = batch_size



        # counters

        self.training_iter = 0
        self.training_count = 0
        self.effective_training_iter = 0
        self.demonstration_training_iter = 0

        # writer
        self.ma_path = runs_name + "/" + "Manager"
        self.writer = SummaryWriter(log_dir=self.ma_path)

    def safe(self, obs, threshold=None):
        """
        Checks if the grid is in a safe state based on the rho threshold.

        Args:
            obs (grid2op.Observation.Observation):
                The current observation of the grid.

        Returns:
            bool: True if all lines are within the rho threshold, False otherwise.
        """
        if threshold is None:
            return np.all(obs.rho <= self.rho_threshold)
        else:
            return np.all(obs.rho <= threshold)

    def compute_main_q_values(self, input, train=True):
        if train:
            self.main_net.train()
        else:
            self.main_net.eval()

        return self.main_net(input)

    def compute_target_q_values(self, input):

        self.target_net.eval()
        with torch.no_grad():
            out = self.target_net(input).detach()

        return out

    def exploit_agent(self, obs):
        obs = self.obs_to_torch(obs).unsqueeze(0)  # torch vector
        q_values = self.compute_main_q_values(obs, train=False)
        return self.playable[torch.argmax(q_values).item()]

    def play_agent(self, obs):
        obs = self.obs_to_torch(obs).unsqueeze(0)  # torch vector
        q_values = self.compute_main_q_values(obs, train=False)
        ac=self.policy.select_action(q_values)
        return self.playable[ac[0]], ac[1]

    def store_experience(self, experience):
        self.buffer.add(experience=experience)

    def store_demonstration(self, demonstration):  # g-a-r-ng-d-nng-nd
        """
        just to remind how the sampling works: s-a-r-ns-d-nng-nd
        """
        demonstration = (
            demonstration[0],
            self.playable.index(demonstration[1]),
            *demonstration[2:]
        )

        self.demonstration_buffer.add(demonstration)

        self.buffer.add((demonstration[0], demonstration[1], demonstration[2][0], demonstration[3], demonstration[4]))

    def _training_iteration(self):
        if self.buffer_type == "basic":
            states, actions, rewards, next_states, dones, _, _ = self.buffer.sample(self.batch_size, way="s-a-r-ns-d",
                                                                                    demonstrations=False)
            loss, _ = self.compute_loss(batch_state=states, batch_action=actions, batch_reward=rewards,
                                        batch_next_state=next_states, batch_done=dones)

        else:
            idxs, batch, weights = self.buffer.sample(self.batch_size, way="s-a-r-ns-d", demonstrations=False)
            states, actions, rewards, next_states, dones, _, _ = batch

            loss, td_error = self.compute_loss(batch_state=states, batch_action=actions, batch_reward=rewards,
                                               batch_next_state=next_states, batch_done=dones, IS_weights=weights)
            self.buffer.update(idxs=idxs, errors=td_error)

        self.writer.add_scalar("Loss", loss.item(), self.training_iter)
        self.update_parameters(loss=loss)

    def _demonstration_training_iteration(self):
        # sample a batch
        states, actions, rewards, next_states, dones, states_tn, dones_tn = self.demonstration_buffer.sample(
            self.batch_size, way="s-a-r-ns-d", demonstrations=True)
        loss, _ = self.compute_demonstration_loss(states=states, actions=actions, rewards=rewards,
                                                  next_states=next_states, dones=dones, states_tn=states_tn,
                                                  dones_tn=dones_tn)

        if self.log_loss: self.writer.add_scalar("Demonstration Loss", loss.item(), self.demonstration_training_iter)

        # update parameters
        self.update_parameters(loss)
        self.demonstration_training_iter += 1

    def update_parameters(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.main_net.parameters(), self.gradient_clipping)
        self.optimizer.step()

    def compute_loss(self, batch_state, batch_action, batch_reward, batch_next_state, batch_done, IS_weights=None):
        ### note the state must already have been processed by the graph
        ### this function returns only the loss, another function for the optimizer
        if IS_weights is None and self.buffer_type == "basic":

            actions = batch_action
            q_values = self.compute_main_q_values(batch_state, train=True)
            q_values = torch.gather(q_values, 1, actions).view(-1)

            with torch.no_grad():
                if self.use_double:
                    best_main_actions = self.compute_main_q_values(batch_next_state, train=False).argmax(dim=1,
                                                                                                         keepdim=True).detach()
                    target_q_values = self.compute_target_q_values(batch_next_state).gather(1, best_main_actions).view(
                        -1)
                else:
                    target_q_values = self.compute_target_q_values(batch_next_state)
                    target_q_values, _ = torch.max(target_q_values, dim=1)

                target_q_values = batch_reward + (1 - batch_done) * self.gamma * target_q_values

            return self.loss(q_values, target_q_values), None
        else:
            actions = batch_action
            q_values = self.compute_main_q_values(batch_state, train=True)
            q_values = torch.gather(q_values, 1, actions).view(-1)

            with torch.no_grad():
                if self.use_double:
                    best_main_actions = self.compute_main_q_values(batch_next_state, train=False).argmax(dim=1,
                                                                                                         keepdim=True).detach()
                    target_q_values = self.compute_target_q_values(batch_next_state).gather(1, best_main_actions).view(
                        -1)
                else:
                    target_q_values = self.compute_target_q_values(batch_next_state)
                    target_q_values, _ = torch.max(target_q_values, dim=1)

                target_q_values = batch_reward + (1 - batch_done) * self.gamma * target_q_values

            TD_errors = target_q_values - q_values
            loss = (torch.tensor(IS_weights, device=self.device) * TD_errors.pow(2)).sum()

            return loss, TD_errors.detach().cpu().numpy()

    def compute_demonstration_loss(self, states, actions, rewards, next_states, dones, states_tn, dones_tn):
        ####### standard DQN loss   Note: only prioritized buffer

        q_values = self.compute_main_q_values(states, train=True)
        batch_size, n_actions = q_values.shape  # number of rows , number of columns of the q_values
        q_values = torch.gather(q_values, 1, actions).view(-1)
        n = rewards.shape[1]  # number of time step considered

        with torch.no_grad():
            if self.use_double:
                best_main_actions = self.compute_main_q_values(next_states, train=False).argmax(dim=1,
                                                                                                keepdim=True).detach()
                target_q_values = self.compute_target_q_values(next_states).gather(1, best_main_actions).view(-1)
            else:
                target_q_values = self.compute_target_q_values(next_states)
                target_q_values, _ = torch.max(target_q_values, dim=1)

            target_q_values = (1 - dones) * self.gamma * target_q_values
        target_q_values = target_q_values + rewards[:, 0]

        single_step_loss = self.loss(q_values, target_q_values)

        with torch.no_grad():
            if self.use_double:
                best_main_actions = self.compute_main_q_values(states_tn, train=False).argmax(dim=1,
                                                                                              keepdim=True).detach()
                target_q_values = self.compute_target_q_values(states_tn).gather(1, best_main_actions).view(-1)
            else:
                target_q_values = self.compute_target_q_values(states_tn)
                target_q_values, _ = torch.max(target_q_values, dim=1)

            # computation of the n step reward
            exponents = torch.arange(n).float()
            rewards = torch.sum(rewards * self.gamma ** exponents, axis=1)  # shape (batch_size)

            target_q_values = rewards + self.gamma ** n * target_q_values * (1 - dones_tn)

        n_step_loss = self.loss(q_values, target_q_values)

        # supervised loss
        margin = torch.zeros(batch_size, n_actions) + self.margin
        indexes = torch.arange(batch_size)
        margin[indexes, actions.view(-1)] = 0  # place the zeros on the expert actions
        supervised_loss = torch.clip(torch.max(self.compute_main_q_values(states, train=True) + margin) - q_values,
                                     0).mean()

        # regularization loss
        l2_norm = sum(param.pow(2.0).sum() for param in self.main_net.parameters())

        return single_step_loss + self.lambda1 * n_step_loss + self.lambda2 * supervised_loss + self.lambda3 * l2_norm, None

    def learn(self):

        if len(self.buffer) < self.start_training_capacity:
            self.training_iter += self.num_training_iters
            self.training_count += 1
            return

        else:
            for i in range(self.num_training_iters):
                self._training_iteration()
                self.training_iter += 1

            self.training_count += 1
            self.effective_training_iter += self.num_training_iters

        if self.policy_type == "epsilon-greedy":
            self.writer.add_scalar("Epsilon", self.policy.epsilon, self.training_count)

        else:
            self.writer.add_scalar("Temperature", self.policy.temperature, self.training_count)

        self.policy.update()

    def learn_demonstrations(self):
        if len(self.demonstration_buffer) < self.min_demonstrations:
            return
        else:
            for i in range(self.num_training_iters):
                self._demonstration_training_iteration()

    def sync_target(self):
        if self.soft_updates:
            soft_update(self.target_net, self.main_net, self.tau_soft_updates)
        else:
            self.target_net.load_state_dict(self.main_net.state_dict())

    def save_checkpoint(self):
        torch.save({
            'main_net_state_dict': self.main_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, self.ma_path + "/checkpoint.pth")

    def load_checkpoint(self):
        checkpoint = torch.load(self.ma_path + "/checkpoint.pth")
        self.main_net.load_state_dict(checkpoint["main_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


    def check_disconnections(self, obs):
        """
        Checks if there is any disconnected line in the grid.

        Args:
            obs (grid2op.Observation.Observation):
                The current observation of the grid.

        Returns:
            bool: True if there is at least one disconnected line, False otherwise.
        """
        return np.any(obs.rho == 0.0)

    def reconnect_line(self, obs):
        """
        Reconnects a random disconnected line.

        Args:
            obs (grid2op.Observation.Observation):
                The current observation of the grid.

        Returns:
            grid2op.Action.Action: The action to reconnect a randomly selected disconnected line.

        Raises:
            AssertionError: If there are no disconnected lines to reconnect.
        """
        line = np.where(obs.rho == 0.0)[0]
        assert len(line) > 0
        line_id = random.choice(line)
        return self.act_space.reconnect_powerline(line_id=line_id, bus_or=0, bus_ex=0)

    def do_nothing(self):
        """
        Returns a "do-nothing" action.

        Returns:
            grid2op.Action.Action: The "do-nothing" action.
        """
        return self.act_space({})

    def obs_to_torch(self, obs):
        return torch.tensor(np.concatenate([obs.rho, obs.a_or, obs.a_ex, obs.topo_vect], axis=0), device=self.device,
                            dtype=torch.float32)


def soft_update(target_network, main_network, tau):
    """
    Perform a soft update of the target network.

    Args:
        target_network (nn.Module): The target network (to be updated).
        main_network (nn.Module): The main network (providing new parameters).
        tau (float): Soft update coefficient (0 < tau <= 1).
    """
    with torch.no_grad():  # Ensures no gradients are computed
        for target_param, main_param in zip(target_network.parameters(), main_network.parameters()):
            target_param.data.copy_(tau * main_param.data + (1.0 - tau) * target_param.data)
