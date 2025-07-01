from ..DeepL.GraphNN import Graph_processor_lines # type: ignore
import torch

class GP_Manager:
    def __init__(self, grid2openv, input_dim , n_mp, type ,dropout, batch_size, lr, weight_decay, gradient_clipping, soft_updates, tau_soft_updates=None,  device="cpu"):
            #self.grid2openv=grid2openv  otherwise it become unpickable and other measures are requires
            self.input_dim=input_dim
            self.n_mp=n_mp
            self.batch_size=batch_size
            self.dropout=dropout
            self.lr=lr
            self.type=type
            self.weight_decay=weight_decay
            self.gradient_clipping=gradient_clipping
            self.device=device
            self.soft_updates=soft_updates
            self.tau_soft_updates=tau_soft_updates
             
            # graph processors
            self.main_gp=Graph_processor_lines(environment=grid2openv,input_dim=input_dim,n_mp=n_mp,type=type , dropout=dropout,batch_size=batch_size,device=device)
            self.target_gp=Graph_processor_lines(environment=grid2openv,input_dim=input_dim,n_mp=n_mp,type=type,dropout=dropout,batch_size=batch_size,device=device)

            self.target_gp.load_state_dict(self.main_gp.state_dict())
            self.target_gp.eval()

            # optimizer
            self.optimizer = torch.optim.Adam(self.main_gp.parameters(), lr=lr, weight_decay=weight_decay)



    def restore_optimizer(self, state_dict):
        assert not hasattr(self , "optimizer") #assert self.optimizer is None
        opt=torch.optim.Adam(self.main_gp.parameters())
        opt.load_state_dict(state_dict=state_dict)
        setattr(self , "optimizer" ,  opt )
        assert self.optimizer.state_dict()== state_dict

    def process_batch(self, graph, train, target):
        """
        train and target are bool
        """
        if target:
            with torch.no_grad():
                out=self.target_gp.forward(graph.x,graph.edge_index).detach()

            return out

        else:
             if train:
                self.main_gp.train()
                return self.main_gp.forward(graph.x,graph.edge_index)
             else:
                self.main_gp.eval()
                return self.main_gp.forward(graph.x,graph.edge_index)

    
    def process_graph(self,graph, train, target):
        """
        train and target are bool
        """
        if target:
            with torch.no_grad():
                out=self.target_gp.process(graph).detach()
            return out

        else:
             if train:
                self.main_gp.train()
                return self.main_gp.process(graph)
             else:
                self.main_gp.eval()
                return self.main_gp.process(graph)
             

    def sync_target(self):
        """
        Sync between the main graph processor and the target one

        Returns:
            None
        """
        if self.soft_updates:
            soft_update(self.target_gp, self.main_gp, self.tau_soft_updates)
        else:
            self.target_gp.load_state_dict(self.main_gp.state_dict())

    
    def save_checkpoint(self, path):
        torch.save({
        'main_gp_state_dict': self.main_gp.state_dict(),
        'target_gp_state_dict': self.target_gp.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict(),
        }, path+"/gp_checkpoint.pth")


    def load_checkpoint(self,path):
        checkpoint = torch.load(path+"/gp_checkpoint.pth")
        self.main_gp.load_state_dict(checkpoint["main_gp_state_dict"])
        self.target_gp.load_state_dict(checkpoint["target_gp_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


            
            
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


