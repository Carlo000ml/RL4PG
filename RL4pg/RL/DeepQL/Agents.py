import torch
from torch.utils.tensorboard import SummaryWriter
from ..DeepQL.Policy import EpsilonGreedyPolicy , BoltzmannPolicy # type: ignore
from ..DeepQL.Estimator_Manager import Estimator_Manager # type: ignore
from ...Graph_Processing.GP_Manager import GP_Manager # type: ignore
from ..ReplyBuffers import BasicReplayBuffer, Basic_PrioritizedReplayBuffer
import pickle
from ...utils import action_masking, bad_action_masking
class DQN_Agent:

    def __init__(self, 
                action_space,
                agent_id, 
                runs_name, 
                policy_type="epsilon-greedy",
                policy_kargs={},
                Q_value_estimator_kargs={},
                gp_manager=None,
                replay_buffer_kargs={},
                buffer_type="basic",
                start_training_capacity=1500,
                num_training_iters=100,
                log_loss=True,
                bad_action_masking=True,
                learn_from_demonstrations=False,
                min_demonstrations=500,
                rew_shape=True,
                device='cpu'):
        assert policy_type in ["epsilon-greedy" , "boltzmann"]
        if len(action_space)==0: return
        self.action_space = action_space
        self.action_space_dim=len(action_space)
        self.agent_id = agent_id
        self.runs_name = runs_name
        self.policy_type=policy_type
        self.path=runs_name+"/"+str(agent_id)
        self.writer = SummaryWriter(log_dir=self.path)
        
        self.start_training_capacity = start_training_capacity
        self.num_training_iters=num_training_iters
        self.device = device
        self.log_loss=log_loss
        self.bad_action_masking=bad_action_masking

        self.learn_from_demonstrations=learn_from_demonstrations
        self.min_demonstrations=min_demonstrations

        self.rew_shape=rew_shape


        # policy 
        if policy_type=="epsilon-greedy": 
            policy_kargs["action_space_dim"]=self.action_space_dim
            self.policy=EpsilonGreedyPolicy(**policy_kargs)
        else: 
            policy_kargs["action_space_dim"]=self.action_space_dim
            self.policy=BoltzmannPolicy(**policy_kargs)

        # q-value estimator
        Q_value_estimator_kargs["writer"]=self.writer
        Q_value_estimator_kargs["device"]=self.device
        Q_value_estimator_kargs["Q_estimator_params"]["output_dim"]=self.action_space_dim
        self.estimator_manager=Estimator_Manager(**Q_value_estimator_kargs)


        # gp manager
        self.gp_manager=gp_manager


        # replay buffer
        assert buffer_type in [ "basic" , "prioritized"]
        self.buffer_type=buffer_type
        replay_buffer_kargs["device"]=self.device
        if self.buffer_type=="basic":
            self.buffer=BasicReplayBuffer(**replay_buffer_kargs)
            self.demonstration_buffer=BasicReplayBuffer(**replay_buffer_kargs)
        elif self.buffer_type=="prioritized":
            self.buffer=Basic_PrioritizedReplayBuffer(**replay_buffer_kargs)
            self.demonstration_buffer=Basic_PrioritizedReplayBuffer(**replay_buffer_kargs)



        # time steps
        self.global_training_iter=0
        self.effective_training_iter=0
        self.training_count=0
        self.global_training_count=0
        self.global_demonstration_training_iter=0
        self.effective_demonstration_training_iter=0




    def exploit(self,graph, current_obs=None):
        # processing the graph using the main graph nn, in evaluation mode, and selecting the line conrresponding to the agent
        obs=self.gp_manager.process_graph(graph , train=False, target=False)[self.agent_id].unsqueeze(0)
        #computing q values using the main network in evaluation mode
        q_values=self.estimator_manager.compute_main_q_values(obs, train=False)

        # action masking
        fake_indexes=action_masking(self.action_space,current_obs).to(self.device)
        if self.bad_action_masking:
            bad_indexes=bad_action_masking(self.action_space,current_obs).to(self.device)
            fake_indexes=torch.cat([fake_indexes,bad_indexes])

        q_values[0][fake_indexes]=-1e9

        # select the optimal action according to a greedy policy
        index=torch.argmax(q_values).item()
        return self.action_space[index]
    

    def act(self,graph, current_obs=None):
        obs=self.gp_manager.process_graph(graph , train=False, target=False)[self.agent_id].unsqueeze(0)
        q_values=self.estimator_manager.compute_main_q_values(obs, train=False)
        # using the policy and action masking
        fake_indexes=action_masking(self.action_space,current_obs).to(self.device)
        if self.bad_action_masking:
            bad_indexes=bad_action_masking(self.action_space,current_obs).to(self.device)
            fake_indexes=torch.cat([fake_indexes,bad_indexes])
        q_values[0][fake_indexes]=-1e9

        index, exploration=self.policy.select_action(q_values)
        return self.action_space[index], exploration
    
    def _standard_training_iteration(self):
        # sample a batch
        state, action, reward, next_state, done , _, _ = self.buffer.sample(self.gp_manager.batch_size,way="g-a-r-ng", demonstrations=False)
        # process in batch using the main and the target gp and estimators
        main_obs=self.gp_manager.process_batch(state, train=True, target=False)[self.agent_id]
        target_obs=self.gp_manager.process_batch(next_state, train=False, target=True)[self.agent_id]   # useless the train=False but you never know

        # compute loss
        if self.rew_shape:
            loss=self.estimator_manager.compute_loss_BSRS(main_obs, action, reward, target_obs, done )
        else:
            loss=self.estimator_manager.compute_loss(main_obs, action, reward, target_obs, done )

        if self.log_loss: self.writer.add_scalar("Loss",loss.item() , self.global_training_iter)

        # update parameters
        self.update_parameters(loss)
    
        # perform the step updates
        self.global_training_iter+=1


    def _standard_demonstration_training_iteration(self):
        # sample a batch
        batch_state_t, batch_action_t, batch_rewards_t_n, batched_states_t_1, batch_done_t,batch_state_t_n, batch_done_t_n = self.demonstration_buffer.sample(self.gp_manager.batch_size, way="g-a-r-ng-d", demonstrations=True)
        # process in batch using the main and the target gp and estimators
        main_obs=self.gp_manager.process_batch(batch_state_t, train=True, target=False)[self.agent_id]
        target_obs_t_1=self.gp_manager.process_batch(batched_states_t_1, train=False, target=True)[self.agent_id]   # useless the train=False but you never know
        target_obs_t_n=self.gp_manager.process_batch(batch_state_t_n, train=False, target=True)[self.agent_id]   # useless the train=False but you never know


        # compute loss
        loss=self.estimator_manager.compute_demonstration_loss(batch_state_t=main_obs,
                                                        batch_action_t=batch_action_t,
                                                        batch_rewards_t_n=batch_rewards_t_n,
                                                        batch_done_t=batch_done_t,
                                                        batched_states_t_1=target_obs_t_1,
                                                        batch_state_t_n=target_obs_t_n,
                                                        batch_done_t_n=batch_done_t_n
                                                        )

        if self.log_loss: self.writer.add_scalar("Demonstration Loss",loss.item() , self.global_demonstration_training_iter)

        # update parameters
        self.update_parameters(loss)
    
        # perform the step updates
        self.global_demonstration_training_iter+=1
    
    def standard_learn(self):

        if len(self.buffer)< self.start_training_capacity:
            self.global_training_iter+=self.num_training_iters
            self.global_training_count+=1
            return
        
        else:
            for _ in range(self.num_training_iters):
                self._standard_training_iteration()
                
            self.training_count+=1
            self.global_training_count+=1
            self.effective_training_iter+=self.num_training_iters
            
            # update the policy
            if self.policy_type=="epsilon-greedy":
                self.writer.add_scalar("Epsilon",self.policy.epsilon , self.global_training_count)

            else:
                self.writer.add_scalar("Temperature",self.policy.temperature , self.global_training_count)

            self.policy.update()

    def standard_learn_demonstrations(self):
        
        if len(self.demonstration_buffer)< self.min_demonstrations:
            self.global_demonstration_training_iter+=self.num_training_iters
            return
        
        else:

            for _ in range(self.num_training_iters):
                self._standard_demonstration_training_iteration()

            self.effective_demonstration_training_iter+=self.num_training_iters

       
        return

    def _prioritized_training_iteration(self):
        # sample a batch
        idxs, batch, weights = self.buffer.sample(self.gp_manager.batch_size, way="g-a-r-ng-d", demonstrations=False)

        state, action, reward, next_state, done, _, _ = batch
        # process in batch using the main and the target gp and estimators
        main_obs=self.gp_manager.process_batch(state, train=True, target=False)[self.agent_id]
        target_obs=self.gp_manager.process_batch(next_state, train=False, target=True)[self.agent_id]   # useless the train=False but you never know

        # compute loss
        if self.rew_shape:
            loss, td_error =self.estimator_manager.compute_loss_with_TD_errors_BSRS(batch_state=main_obs, batch_action=action, batch_reward=reward, batch_next_state=target_obs, batch_done=done , IS_weights=weights)
        else:
            loss, td_error =self.estimator_manager.compute_loss_with_TD_error(batch_state=main_obs, batch_action=action, batch_reward=reward, batch_next_state=target_obs, batch_done=done , IS_weights=weights)

        if self.log_loss: self.writer.add_scalar("Loss",loss.item() , self.global_training_iter)

        # update parameters
        self.update_parameters(loss)

        # update priorities
        self.buffer.update(idxs=idxs,errors=td_error)
    
        # perform the step updates
        self.global_training_iter+=1




    def _prioritized_demonstrations_training_iteration(self):
        # sample a batch
        idxs, batch, weights = self.demonstration_buffer.sample(self.gp_manager.batch_size, way="g-a-r-ng-d", demonstrations=True)

        batch_state_t, batch_action_t, batch_rewards_t_n, batched_states_t_1, batch_done_t,batch_state_t_n, batch_done_t_n  = batch
        # process in batch using the main and the target gp and estimators
        main_obs=self.gp_manager.process_batch(batch_state_t, train=True, target=False)[self.agent_id]
        target_obs_t_1=self.gp_manager.process_batch(batched_states_t_1, train=False, target=True)[self.agent_id]   # useless the train=False but you never know
        target_obs_t_n=self.gp_manager.process_batch(batch_state_t_n, train=False, target=True)[self.agent_id]   #  # useless the train=False but you never know
        # compute loss
        loss, td_error =self.estimator_manager.compute_demonstration_loss_TD_error(batch_state_t=main_obs,
                                                                                batch_action_t=batch_action_t,
                                                                                batch_rewards_t_n=batch_rewards_t_n,
                                                                                batch_done_t=batch_done_t,
                                                                                batched_states_t_1=target_obs_t_1,
                                                                                batch_state_t_n=target_obs_t_n,
                                                                                batch_done_t_n=batch_done_t_n,
                                                                                IS_weights=weights)

        if self.log_loss: self.writer.add_scalar("Demonstration Loss",loss.item() , self.global_demonstration_training_iter)

        # update parameters
        self.update_parameters(loss)

        # update priorities
        self.demonstration_buffer.update(idxs=idxs,errors=td_error)
    
        # perform the step updates
        self.global_demonstration_training_iter+=1


    def prioritized_learn(self):

        if len(self.buffer)< self.start_training_capacity:
            self.global_training_iter+=self.num_training_iters
            self.global_training_count+=1
            return
        
        else:

            for _ in range(self.num_training_iters):
                self._prioritized_training_iteration()
                
            
            self.training_count+=1
            self.global_training_count+=1
            self.effective_training_iter+=self.num_training_iters
            
            # update the policy
            if self.policy_type=="epsilon-greedy":
                self.writer.add_scalar("Epsilon",self.policy.epsilon , self.global_training_count)

            else:
                self.writer.add_scalar("Temperature",self.policy.temperature , self.global_training_count)

            self.policy.update()

    def prioritized_learn_demonstrations(self):

        if len(self.demonstration_buffer)< self.min_demonstrations:
            self.global_demonstration_training_iter+=self.num_training_iters
            return
        
        else:

            for _ in range(self.num_training_iters):
                self._prioritized_demonstrations_training_iteration()

            self.effective_demonstration_training_iter+=self.num_training_iters

       
        return

        

    def learn(self):
        if self.buffer_type=="basic": self.standard_learn()
        if self.buffer_type=="prioritized": self.prioritized_learn()

    def learn_demonstrations(self):
        if self.buffer_type=="basic": self.standard_learn_demonstrations()
        if self.buffer_type=="prioritized": self.prioritized_learn_demonstrations()

    def update_parameters(self, loss):
        self.estimator_manager.optimizer.zero_grad()
        self.gp_manager.optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_value_(self.estimator_manager.main_net.parameters(), self.estimator_manager.gradient_clipping)
        torch.nn.utils.clip_grad_value_(self.gp_manager.main_gp.parameters(), self.gp_manager.gradient_clipping)

        self.estimator_manager.optimizer.step()
        self.gp_manager.optimizer.step()

    def save_estimator_checkpoint(self):
        self.estimator_manager.save_checkpoint()

    def load_estimator_checkpoint(self):
        self.estimator_manager.load_checkpoint()



    def store_experience(self,  experience): # graph, action, reward, next_graph, done):
        """
        Stores an experience in the replay buffer.

        Args: 
            experience: whatever
        """
       # action=self.grid2op_to_torch(action)

        self.buffer.add(experience)
        
    def store_demonstration(self,demonstration): # g-a-r-ng-d-nng-nd - rews-rewns
        """
        just to remind how the sampling works: g-a-r-ng-d-nng-nd  states, actions, rewards, done_t,state_t_1, state_t_n, done_n
        """
        self.demonstration_buffer.add(demonstration)
       
        self.buffer.add((demonstration[0] ,demonstration[1] , demonstration[2][0], demonstration[4] ,demonstration[3]))


    def restore_writer(self):
        assert not hasattr(self, "writer")
        writer = SummaryWriter(log_dir=self.path)
        setattr(self, "writer" , writer)
        assert hasattr(self,"writer")


    def grid2op_to_torch(self,grid2op_action):
        """
        Converts a grid2op action to a PyTorch tensor.

        Args:
            grid2op_action (grid2op.Action.Action): Action to convert.

        Returns:
            torch.Tensor: Action index as a tensor.
        """
        return torch.tensor(self.action_space.index(grid2op_action), dtype=torch.long , device=self.device)
    


    def __getstate__(self):
        """
        Called when pickling the object. Exclude unpicklable attributes like `SummaryWriter`.
        """
        state = self.__dict__.copy()  # Get all attributes of the object
        # Remove unpicklable attributes
        if "writer" in state:
            del state["writer"]  # Exclude SummaryWriter

        if "estimator_manager" in state and hasattr(state["estimator_manager"], "optimizer"):
            del state["estimator_manager"].optimizer

        if "estimator_manager" in state and hasattr(state["estimator_manager"], "writer"):
            del state["estimator_manager"].writer

        if "gp_manager" in state and hasattr(state["gp_manager"], "optimizer"):
            del state["gp_manager"].optimizer
            
        if "gp_manager" in state and hasattr(state["gp_manager"], "grd2openv"):
            del state["gp_manager"].grd2openv
            
        return state

    def __setstate__(self, state):
        """
        Called when unpickling the object. Restore any attributes that need to be reinitialized.
        """
        self.__dict__.update(state)  # Restore the attributes
        # Reinitialize the unpicklable attributes
        self.writer = SummaryWriter(log_dir=self.path)

    def save(self):
        """
        Save the DQNAgent object to a file.

        Args:
            path (str): File path where the object will be saved (without extension).
        """

        path=self.runs_name+f'/Agent_{self.agent_id}.pkl'
        with open(path , "wb") as f:
            pickle.dump(self,f)


    @staticmethod
    def load(path):
        """
        Load a DQNAgent object from a file.

        Args:
            path (str): File path where the object is saved (including .pkl extension).

        Returns:
            DQNAgent: The loaded DQNAgent object.
        
        Usage:
            loaded_agent = DQNLine_Agent.load("dqn_agent.pkl")
        """
        with open(path, "rb") as f:
            agent = pickle.load(f)
        print(f"Agent loaded from {path}")
        return agent



    def close_writer(self):
        """
        Closes the TensorBoard writer.
        """
        self.writer.close()