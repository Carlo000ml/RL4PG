import torch
from RL4pg.RL.DeepQL.Q_estimators import DQNetwork,DuelingQNetwork # type: ignore
import torch.nn as nn
class Estimator_Manager:
    def __init__(self, 
                lr=1e-3, 
                weight_decay=1e-4,
                gamma=0.999,
                loss_type="MSE",
                delta_huber=1,
                margin=1,
                lambda1=0.2,
                lambda2=0.2,
                lambda3=0.2,
                Q_estimator_net_type="Dueling",
                use_double=False,
                Q_estimator_params=None,
                target_update_freq=100,
                device='cpu',
                log_loss=False,
                gradient_clipping=torch.inf,
                soft_updates=False,
                tau_soft_updates=0.5,
                eta=None,
                writer=None               
                ):
        self.loss_type=loss_type
        self.lr=lr
        self.weight_decay=weight_decay
        self.gamma=gamma
        self.Q_estimator_net_type=Q_estimator_net_type
        self.use_double=use_double
        self.Q_estimator_params=Q_estimator_params
        self.target_update_freq=target_update_freq
        self.device=device
        self.log_loss=log_loss
        self.gradient_clipping=gradient_clipping
        self.soft_updates=soft_updates
        self.tau_soft_updates=tau_soft_updates
        self.writer=writer
        self.path=writer.log_dir
        self.margin=margin
        self.lambda1=lambda1
        self.lambda2=lambda2
        self.lambda3=lambda3
        if eta:
            self.eta=eta
        else:
            self.eta = ((1 - gamma) / (1 + gamma)) / 2
        
        # estimators
        assert Q_estimator_net_type in ["Dueling" , "Simple"]
        assert loss_type in ["MSE" , "Huber"]
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

        # loss
        if loss_type=="MSE":
            self.loss = nn.MSELoss()
        else:
            self.loss=nn.HuberLoss(delta=delta_huber)


    def save_checkpoint(self):
        torch.save({
        'main_net_state_dict': self.main_net.state_dict(),
        'target_net_state_dict': self.target_net.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict(),
        }, self.path+"/checkpoint.pth")


    def load_checkpoint(self):
        checkpoint = torch.load(self.path+"/checkpoint.pth")
        self.main_net.load_state_dict(checkpoint["main_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])




    def restore_optimizer(self, state_dict):
         ##print("the optimizer is: ", self.optimizer)
        assert not hasattr(self , "optimizer") #assert self.optimizer is None
        opt=torch.optim.Adam(self.main_net.parameters())
        opt.load_state_dict(state_dict=state_dict)
        setattr(self , "optimizer" ,  opt )
        assert self.optimizer.state_dict()== state_dict

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

        


    def compute_loss(self, batch_state, batch_action, batch_reward, batch_next_state, batch_done ):
        ### note the state must already have been processed by the graph
        ### this function returns only the loss, another function for the optimizer

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

                target_q_values= (1-batch_done) * self.gamma * target_q_values

            target_q_values=target_q_values+batch_reward

            return self.loss(q_values,target_q_values)
    

    
    def compute_loss_BSRS(self, batch_state, batch_action, batch_reward, batch_next_state, batch_done ):
        # compute q values for st:
        actions=batch_action
        q_values=self.compute_main_q_values(batch_state, train=True)
        q_values=torch.gather(q_values, 1, actions).view(-1)
        #  compute the state-value for st
        with torch.no_grad():
            if self.use_double:
                best_main_actions=self.compute_main_q_values(batch_state, train=False).argmax(dim=1, keepdim=True).detach()
                value_st=self.compute_target_q_values(batch_state).gather(1, best_main_actions).view(-1)
            else:
                value_st=self.compute_target_q_values(batch_state)
                value_st, _ = torch.max(value_st, dim=1)

        # compute the state-value for st+1
        with torch.no_grad():
                if self.use_double:
                    best_main_actions=self.compute_main_q_values(batch_next_state, train=False).argmax(dim=1, keepdim=True).detach()
                    value_st1=self.compute_target_q_values(batch_next_state).gather(1, best_main_actions).view(-1)
                else:
                    value_st1=self.compute_target_q_values(batch_next_state)
                    value_st1, _ = torch.max(value_st1, dim=1)

                target_q_values= batch_reward-self.eta* value_st   + (1-batch_done) * self.gamma * (1+self.eta) * value_st1

        return self.loss(q_values,target_q_values)
        
    

    def compute_loss_with_TD_error(self, batch_state, batch_action, batch_reward, batch_next_state, batch_done, IS_weights):
            """ 
            Computing the loss for the PER version. 
            The implementation follows the original DeepMind implementation, the loss is the sum of the weighted batch losses.
            """

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

                target_q_values= (1-batch_done) * self.gamma * target_q_values
            target_q_values= target_q_values + batch_reward

            TD_errors=target_q_values-q_values
            loss = (torch.tensor(IS_weights, device=self.device) * TD_errors.pow(2)).sum()


            return loss , TD_errors.detach().cpu().numpy()
    
    def compute_loss_with_TD_errors_BSRS(self, batch_state, batch_action, batch_reward, batch_next_state, batch_done, IS_weights):
        # compute q values for st:
        actions=batch_action
        q_values=self.compute_main_q_values(batch_state, train=True)
        q_values=torch.gather(q_values, 1, actions).view(-1)
        #  compute the state-value for st
        with torch.no_grad():
            if self.use_double:
                best_main_actions=self.compute_main_q_values(batch_state, train=False).argmax(dim=1, keepdim=True).detach()
                value_st=self.compute_target_q_values(batch_state).gather(1, best_main_actions).view(-1)
            else:
                value_st=self.compute_target_q_values(batch_state)
                value_st, _ = torch.max(value_st, dim=1)

        # compute the state-value for st+1
        with torch.no_grad():
                if self.use_double:
                    best_main_actions=self.compute_main_q_values(batch_next_state, train=False).argmax(dim=1, keepdim=True).detach()
                    value_st1=self.compute_target_q_values(batch_next_state).gather(1, best_main_actions).view(-1)
                else:
                    value_st1=self.compute_target_q_values(batch_next_state)
                    value_st1, _ = torch.max(value_st1, dim=1)

                target_q_values= batch_reward-self.eta* value_st   + (1-batch_done) * self.gamma * (1+self.eta) * value_st1

        TD_errors=target_q_values-q_values
        loss = (torch.tensor(IS_weights, device=self.device) * TD_errors.pow(2)).sum()

        return loss , TD_errors.detach().cpu().numpy()

    

    def compute_demonstration_loss(self, batch_state_t, batch_action_t, batch_rewards_t_n, batch_done_t, batched_states_t_1,batch_state_t_n, batch_done_t_n):
            
            ##### how the batch reward is returned here? what type and what shape?  A tensor of shape (batch_size, n)
            actions=batch_action_t
            q_values=self.compute_main_q_values(batch_state_t, train=True)
            batch_size,n_actions=q_values.shape  # number of rows , number of columns of the q_values
            q_values=torch.gather(q_values, 1, actions).view(-1)
            n=batch_rewards_t_n.shape[1]

            with torch.no_grad():
                if self.use_double:
                    best_main_actions=self.compute_main_q_values(batched_states_t_1, train=False).argmax(dim=1, keepdim=True).detach()
                    target_q_values=self.compute_target_q_values(batched_states_t_1).gather(1, best_main_actions).view(-1)
                else:
                    target_q_values=self.compute_target_q_values(batched_states_t_1)
                    target_q_values, _ = torch.max(target_q_values, dim=1)

                target_q_values= (1-batch_done_t) * self.gamma * target_q_values
            target_q_values= target_q_values + batch_rewards_t_n[:,0]

            single_step_loss=self.loss(q_values,target_q_values)


            with torch.no_grad():
                if self.use_double:
                    best_main_actions=self.compute_main_q_values(batch_state_t_n, train=False).argmax(dim=1, keepdim=True).detach()
                    target_q_values=self.compute_target_q_values(batch_state_t_n).gather(1, best_main_actions).view(-1)
                else:
                    target_q_values=self.compute_target_q_values(batch_state_t_n)
                    target_q_values, _ = torch.max(target_q_values, dim=1)

                #computation of the n step reward
                exponents=torch.arange(n).float()
                rewards=torch.sum(batch_rewards_t_n* self.gamma**exponents, axis=1)  # shape (batch_size)

                target_q_values= rewards+ self.gamma**n  * target_q_values * (1-batch_done_t_n)

            n_step_loss=self.loss(q_values,target_q_values)

            # supervised loss
            margin= torch.zeros(batch_size, n_actions) + self.margin
            indexes=torch.arange(batch_size)
            margin[indexes,batch_action_t.view(-1)]=0  #  place the zeros on the expert actions
            supervised_loss=torch.clip(torch.max(self.compute_main_q_values(batch_state_t, train=True)+ margin ) - q_values,0).mean()

            # regularization loss
            l2_norm = sum(param.pow(2.0).sum() for param in self.main_net.parameters())





            return single_step_loss+self.lambda1*n_step_loss+self.lambda2* supervised_loss+self.lambda3* l2_norm
    
    def compute_demonstration_loss_TD_error(self,batch_state_t, batch_action_t, batch_rewards_t_n, batch_done_t, batched_states_t_1,batch_state_t_n, batch_done_t_n, IS_weights):
            """ 
            Computing the loss for the PER version. 
            The implementation follows the original DeepMind implementation, the loss is the sum of the weighted batch losses.
            """

            ##### how the batch reward is returned here? what type and what shape?  A tensor of shape (batch_size, n)
            actions=batch_action_t
            q_values=self.compute_main_q_values(batch_state_t, train=True)
            batch_size,n_actions=q_values.shape  # number of rows , number of columns of the q_values
            q_values=torch.gather(q_values, 1, actions).view(-1)
            n=batch_rewards_t_n.shape[1]

            with torch.no_grad():
                if self.use_double:
                    best_main_actions=self.compute_main_q_values(batched_states_t_1, train=False).argmax(dim=1, keepdim=True).detach()
                    target_q_values=self.compute_target_q_values(batched_states_t_1).gather(1, best_main_actions).view(-1)
                else:
                    target_q_values=self.compute_target_q_values(batched_states_t_1)
                    target_q_values, _ = torch.max(target_q_values, dim=1)

                target_q_values=  (1-batch_done_t) * self.gamma * target_q_values
            target_q_values=target_q_values + batch_rewards_t_n[:,0]

            TD_errors=target_q_values-q_values
            single_step_loss = (torch.tensor(IS_weights, device=self.device) * TD_errors.pow(2)).sum()


            with torch.no_grad():
                if self.use_double:
                    best_main_actions=self.compute_main_q_values(batch_state_t_n, train=False).argmax(dim=1, keepdim=True).detach()
                    target_q_values=self.compute_target_q_values(batch_state_t_n).gather(1, best_main_actions).view(-1)
                else:
                    target_q_values=self.compute_target_q_values(batch_state_t_n)
                    target_q_values, _ = torch.max(target_q_values, dim=1)

                #computation of the n step reward
                exponents=torch.arange(n).float()
                rewards=torch.sum(batch_rewards_t_n* self.gamma**exponents, axis=1)  # shape (batch_size)

                target_q_values= rewards+ self.gamma**n  * target_q_values * (1-batch_done_t_n)

            TD_n_error=target_q_values-q_values

            n_step_loss = (torch.tensor(IS_weights, device=self.device) * TD_n_error.pow(2)).sum()


            # supervised loss
            margin= torch.zeros(batch_size, n_actions) + self.margin
            indexes=torch.arange(batch_size)
            margin[indexes,batch_action_t.view(-1)]=0  #  place the zeros on the expert actions
            supervised_loss=torch.clip(torch.max(self.compute_main_q_values(batch_state_t, train=True)+ margin ) - q_values, 0).mean()

            # regularization loss
            l2_norm = sum(param.pow(2.0).sum() for param in self.main_net.parameters())


            return single_step_loss+self.lambda1*n_step_loss+self.lambda2* supervised_loss+self.lambda3* l2_norm , TD_errors.detach().cpu().numpy()





    
        

    def sync_target(self):
        if self.soft_updates:
            soft_update(self.target_net, self.main_net, self.tau_soft_updates)
        else:
            self.target_net.load_state_dict(self.main_net.state_dict())





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