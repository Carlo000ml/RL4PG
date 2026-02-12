import torch
from RL4pg.DeepL.Models import CustomGCN , CustomGNN# type: ignore

class Graph_processor_lines(torch.nn.Module):
 
    def __init__(self, environment , input_dim:int, n_mp:int, type: str, dropout:int, activation='relu' , batch_size=64 ,  device='cpu'):
        """ 
        Graph Processor module. Manages the Graph processing of a graph or of a batch of graphs, here we assume the message passing does not change the shape of the input
        """
    
 
        super(Graph_processor_lines, self).__init__()
        assert type in ["GCN" , "GAT" , "GraphSAGE" , "GIN"]
        self.gcn =CustomGNN(type=type,in_channels=input_dim,hidden_channels=input_dim, out_channels=input_dim,  num_layers=n_mp , dropout=dropout, activation=activation ).to(device)

        self.input_dim=input_dim

        self.n_line=environment.n_line
        
        self.batch_size=batch_size


    def process(self, graph ):

        return self.gcn(graph.x , graph.edge_index)
    
    def forward(self,x, edge_index):   # for the update of the NN in batch

        gcn_out = self.gcn(x, edge_index)   # n_nodes * batch_size

        return gcn_out.view(self.batch_size,self.n_line,self.input_dim).permute(1,0,2)
    
    def process_multigraph(self,multigraph):
        # number of graphs
        nb=len(multigraph.ptr)

        # process it
        gcn_out=self.gcn(multigraph.x,multigraph.edge_index)

        # reshape it to return the shape (number_of_graphs , n_lines, input_dim)
        return gcn_out.view(nb,self.n_line,self.input_dim)


class Graph_processor_sub(torch.nn.Module):

    def __init__(self, environment, input_dim: int, n_mp: int, type: str, dropout: int, activation='relu',
                 batch_size=64, device='cpu'):
        """
        Graph Processor module. Manages the Graph processing of a graph or of a batch of graphs, here we assume the message passing does not change the shape of the input
        """

        super(Graph_processor_sub, self).__init__()
        assert type in ["GCN", "GAT", "GraphSAGE", "GIN"]
        self.gcn = CustomGNN(type=type, in_channels=input_dim, hidden_channels=input_dim, out_channels=input_dim,
                             num_layers=n_mp, dropout=dropout, activation=activation).to(device)

        self.input_dim = input_dim

        self.n_sub = environment.n_sub

        self.batch_size = batch_size

    def process(self, graph):
        return self.gcn(graph.x, graph.edge_index)

    def forward(self, x, edge_index):  # for the update of the NN in batch

        gcn_out = self.gcn(x, edge_index)  # n_nodes * batch_size

        return gcn_out.view(self.batch_size, self.n_sub, self.input_dim).permute(1, 0, 2)

    def process_multigraph(self, multigraph):
        # number of graphs
        nb = len(multigraph.ptr)

        # process it
        gcn_out = self.gcn(multigraph.x, multigraph.edge_index)

        # reshape it to return the shape (number_of_graphs , n_lines, input_dim)
        return gcn_out.view(nb, self.n_sub, self.input_dim)

    

        