import numpy as np
from ..utils import build_line_action_space,build_line_extremities_sub_indexes, build_sub_action_space
import torch
from ..DeepL.Models import CustomRegressor




class Reward_Converter_Line:
    """
    Class to compute the single line reward as sum of squared margin of its substations' powerlines

    Possible developments: 
            - Include neighbors substations
            - Different reward shapes (e.g. -1 * #disconnected_lines)
    Attributes:
        line_id (int): 
            The ID of the substation for which the rewards are being calculated.
        obs (grid2op.BaseObservation)
    Methods:
        grid2op_to_gym(obs):
            Converts a grid2op observation to a custom reward based on line load and disconnection status.
    """
    def __init__(self, line_id, obs):
        """
        Initializes the reward_Converter for a specific substation.

        Args:
            sub_id (int): 
                The ID of the substation for which the reward converter is being initialized.
            obs (grid2op.Observation.Observation): 
                The initial observation of the environment used to determine the lines associated with the substation.

        Attributes:
            sub_id (int): 
                The ID of the substation.
            lines (numpy.ndarray): 
                An array of line IDs that are connected to the substation. These include:
                - Lines for which the substation is the origin (`line_or_to_subid`).
                - Lines for which the substation is the extremity (`line_ex_to_subid`).
        """
        self.line_id=line_id
        self.sub_0=obs.line_ex_to_subid[line_id]
        self.sub_1=obs.line_or_to_subid[line_id]


        # sub 0 
        substation_is_origin_for_lines=np.where(obs.line_or_to_subid==self.sub_0)[0]
        substation_is_extremity_for_lines=np.where(obs.line_ex_to_subid==self.sub_0)[0]
        self.lines_0=np.hstack([substation_is_origin_for_lines,substation_is_extremity_for_lines])

        # sub 1 
        substation_is_origin_for_lines=np.where(obs.line_or_to_subid==self.sub_1)[0]
        substation_is_extremity_for_lines=np.where(obs.line_ex_to_subid==self.sub_1)[0]
        self.lines=np.unique(np.hstack([substation_is_origin_for_lines,substation_is_extremity_for_lines,self.lines_0]))
        self.n_lines=self.lines.shape[0]


        

    def grid2op_to_gym(self,obs):
        """
        Converts a grid2op observation to a custom reward based on the state of the lines 
        connected to the substation.

        ############ Consider giving a negative reward even if rho>1

        The reward is calculated as follows: (rho is clipped between 0 and 1 -> rho > 1 become 1-> minimum reward for the powerline)
        - If there is any line disconnected-> -1*number_disconnected_lines
        - Otherwise, the reward is the squared sum of the differences between `1` and `rho` for all connected lines.

        Args:
            obs (grid2op.Observation.Observation): 
                The current observation of the environment.

        Returns:
            float: 
                The calculated reward value.
        """

        rho=obs.rho[self.lines]
        if np.any(rho == 0) or np.any(rho>=1):
            #if np.sum(rho)==0: return 0#-8064#-1#-4    # Note, it must be 8064 times larger than the min reward of a "good" action
            return 0#-1 #-0.5#-2
        
        rho=np.clip(obs.rho[self.lines], 0, 1)
        
        return   np.sum(   (1-rho)**2   ) /self.n_lines  #( (   np.sum(   (1-rho)**2   )   )    /self.n_lines - 1)  # (-1+ avarage_margin ) # /4   #  (-1+ avarage_margin )/4
    
    


class Action_Converters:
    def __init__(self,env, line_id):
        self.line_id=line_id
        action_space=build_line_action_space(env,line_id=line_id)
        self.action_space=[a.as_dict() for a in action_space]
        self.line_extremities=build_line_extremities_sub_indexes(env)[line_id]



    def grid2op_to_torch(self,action):
        ac=action.as_dict()
        return torch.tensor(self.action_space.index(ac), dtype=torch.long)



class Action_Converters_sub:
    def __init__(self,env, sub_id):
        self.sub_id=sub_id
        action_space=build_sub_action_space(env,sub_id=sub_id)
        self.action_space=[a.as_dict() for a in action_space]



    def grid2op_to_torch(self,action):
        ac=action.as_dict()
        return torch.tensor(self.action_space.index(ac), dtype=torch.long)



class Reward_decomposer:
    def __init__(self,env ,weight_decay=0.3,lr=1e-8, gradient_clipping=1, gamma=0.999, soft_updates=True, tau_soft=0.2,device="cpu", use_target=True):
        """ Note, the potential based approach has theoretical guarantees to be policy invariant (telescopic terms cancel out). 
        
        Additional note: in potential base approach, the global reward is not normalized so that the margin is in magnitude lower than it in general, so initially negligible"""
        super(Reward_decomposer, self).__init__()
        self.n_agents=env.n_line
        self.device=device
        self.gamma=gamma
        self.gradient_clipping=gradient_clipping
        self.soft_updates=soft_updates
        self.tau_soft_updates=tau_soft
        self.net=CustomRegressor(input_dim=117, output_dim=self.n_agents , hidden_layers=[64] ,use_batch_norm=True).to(device)
        self.use_target=use_target
        if use_target:
            self.target=CustomRegressor(input_dim=117, output_dim=self.n_agents , hidden_layers=[64] ,use_batch_norm=True).to(device)
            self.target.load_state_dict(self.net.state_dict())
            self.target.eval()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)


    def _potential_convert(self,states_t,states_t1, global_rews): 
        if self.use_target:
            with torch.no_grad():
                margin=self.gamma * self.target(states_t1)

            margin=margin-self.net(states_t)
            return (global_rews + margin)
        else:
            margin=self.gamma*self.net(states_t1)-self.net(states_t)
            return global_rews+margin
    
    def sync_target(self):
        if self.use_target:
            if self.soft_updates:
                soft_update(self.target, self.net, self.tau_soft_updates)
            else:
                self.target.load_state_dict(self.net.state_dict())
        else:
            return


    def save_checkpoint(self, path):
        if self.use_target:
            torch.save({
            'main_net_state_dict': self.net.state_dict(),
            'target_net_state_dict': self.target.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }, path+"/rd_checkpoint.pth")
        else:
            torch.save({
            'main_net_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }, path+"/rd_checkpoint.pth")


    def load_checkpoint(self, path):
        checkpoint = torch.load(path+"/checkpoint.pth")
        self.net.load_state_dict(checkpoint["main_net_state_dict"])
        if self.use_target:
            self.target.load_state_dict(checkpoint["target_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])



    

    def convert(self,states_t,states_t1,global_rews):
        return self._potential_convert(states_t,states_t1,global_rews)


    def obs_to_torch(self,obs):
        return torch.tensor(np.concatenate([obs.rho, obs.a_or, obs.a_ex, obs.topo_vect],axis=0), device=self.device, dtype=torch.float32)


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