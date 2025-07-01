import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn.models import GCN,GAT,GraphSAGE,GIN
import torch


class CustomRegressor(nn.Module):
    def __init__(self, input_dim : int , output_dim : int , hidden_layers=None, dropout=0.0, use_batch_norm=False, activation="leaky_relu", seed=42, initialization="kaiming uniform"):
        """
        Custom MLP Regressor with configurable hidden layers, batch normalization, dropout, and activation functions.
        Args:
            input_dim (int): Dimension of the input state.
            output_dim (int): Number of actions (output Q-values).
            hidden_layers (list or None): List of hidden layer sizes (e.g., [64, 64]).
                                         Use `None` or an empty list for 0 hidden layers -> input layer - output layer.
            dropout (float): Dropout rate (default: 0.0).
            use_batch_norm (bool): Whether to use batch normalization (default: False).
            activation (str): Activation function to use ("relu", "gelu", "elu", "tanh", "sigmoid", "leaky_relu" ).
        """
        super(CustomRegressor, self).__init__()
        self.generator = torch.Generator().manual_seed(seed)
        assert(initialization in ["xavier uniform" , " xavier normal" ,"kaiming uniform" ])
        self.initialization=initialization
        layers = []

        # Choose the activation function
        activations = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "elu": nn.ELU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
            "leaky_relu": nn.LeakyReLU
        }
        if activation not in activations:
            raise ValueError(f"Unsupported activation function '{activation}'. Choose from {list(activations.keys())}.")
        activation_fn = activations[activation]
        self.activation=activation

        # Case 1: 0 Hidden Layers
        if not hidden_layers:
            layers.append(nn.Linear(input_dim, output_dim))

        # Case 2: 1 or More Hidden Layers
        else:
            # Input layer
            layers.append(nn.Linear(input_dim, hidden_layers[0]))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_layers[0]))
            layers.append(activation_fn())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            # Hidden layers
            for i in range(1, len(hidden_layers)):
                layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
                if use_batch_norm:
                    layers.append(nn.BatchNorm1d(hidden_layers[i]))
                layers.append(activation_fn())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

            # Output layer (no activation or batch normalization)
            layers.append(nn.Linear(hidden_layers[-1], output_dim))

        self.network = nn.Sequential(*layers)
        #self.initialize()  not called because I am not sure it is properly done

    def initialize(self):
        for l in range(len(self.network)):
            custom_init(self.network[l], generator=self.generator , init_type=self.initialization, nonlinearity=self.activation)

    def forward(self, x):
        return self.network(x)





class CustomGCN(nn.Module):
    def __init__(self, input_dim: int, n_layers=1, activation='leaky_relu', seed=42, initialization="kaiming uniform"):

        """
        Configurable GCN with a given number of message passing operations ( each performed by a different GCN layer).
        Args:
            input_dim (int): Input dimension of node features.
            num_layers (int): Number of message passing operations to apply.
            activation (str): Activation function to use ("relu", "gelu", "elu", "tanh", "sigmoid", "leaky_relu" ).
        """
        super(CustomGCN, self).__init__()
        self.generator = torch.Generator().manual_seed(seed)
        self.initialization=initialization
        # Map activation function strings to classes
        activations = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "elu": nn.ELU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
            "leaky_relu": nn.LeakyReLU,
            "linear": nn.Identity
        }
        if activation not in activations:
            raise ValueError(f"Unsupported activation function '{activation}'. Choose from {list(activations.keys())}.")
        self.activation=activation
        self.n_layers=n_layers
        self.activation_fn = activations[activation]()
        self.layers=[GCNConv(in_channels=input_dim, out_channels=input_dim ,normalize=True ) for i in range(n_layers)]
        self.layers=nn.ModuleList(self.layers)
        #self.initialize()  not called because I am not sure it is properly done


    def initialize(self):
        for l in range(len(self.layers)):
            custom_init(self.layers[l], generator=self.generator, init_type=self.initialization, nonlinearity=self.activation)

    
    def forward(self, x, edge_index):
        # Apply single-layer graph convolution
        for i in range(self.n_layers):
            x=self.layers[i](x,edge_index)
            x=self.activation_fn(x)

        
        return x  # No activation or output transformation (raw embeddings)
    


    

class CustomGNN(torch.nn.Module):
    def __init__(self,type, in_channels,hidden_channels, out_channels,num_layers , dropout, activation="relu" , norm='GraphNorm' ):
        assert type in ["GCN", "GraphSAGE" , "GAT" , "GIN"]
        super(CustomGNN, self).__init__()

        model={"GCN": GCN(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels, num_layers=num_layers, dropout=dropout, act=activation,norm=norm),
                "GraphSAGE": GraphSAGE(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels,num_layers=num_layers,dropout=dropout,act=activation,norm=norm),
                "GAT": GAT(in_channels=in_channels,hidden_channels=hidden_channels, out_channels=out_channels, v2=True, num_layers=num_layers,dropout=dropout, act=activation,norm=norm) , 
                "GIN": GIN(in_channels=in_channels, hidden_channels=hidden_channels,out_channels=out_channels, num_layers=num_layers,dropout=dropout,act=activation,norm=norm)}
        self.gnn=model[type]

    def forward(self, x, edge_index):
        return self.gnn(x,edge_index)


def custom_init(m, generator=None, init_type="uniform", nonlinearity=None):
    if isinstance(m, nn.Linear):
        if init_type=="xavier uniform":  
            nn.init.xavier_uniform_(m.weight, generator=generator)  # Xavier uniform
        if init_type=="xavier normal":
            nn.init.xavier_normal_(m.weight, generator=generator)  # Xavier normal
        if init_type=="kaiming uniform":
            nn.init.kaiming_uniform_(m.weight, generator=generator, nonlinearity=nonlinearity)           

    elif isinstance(m,GCNConv):
        if init_type=="uniform": 
            nn.init.xavier_uniform_(m.lin.weight, generator=generator)  # Xavier nuniform
        if init_type=="xavier normal":
            nn.init.xavier_normal_(m.lin.weight, generator=generator)  # Xavier normal
        if init_type=="kaiming uniform":
            nn.init.kaiming_uniform_(m.lin.weight, generator=generator, nonlinearity=nonlinearity) 





class CustomSoftmax(nn.Module):
    def __init__(self, input_dim : int , output_dim : int , hidden_layers=None, dropout=0.0, use_batch_norm=False, activation="leaky_relu", seed=42, initialization="kaiming uniform"):
        """
        Custom MLP Regressor with configurable hidden layers, batch normalization, dropout, and activation functions.
        Args:
            input_dim (int): Dimension of the input state.
            output_dim (int): Number of actions (output Q-values).
            hidden_layers (list or None): List of hidden layer sizes (e.g., [64, 64]).
                                         Use `None` or an empty list for 0 hidden layers -> input layer - output layer.
            dropout (float): Dropout rate (default: 0.0).
            use_batch_norm (bool): Whether to use batch normalization (default: False).
            activation (str): Activation function to use ("relu", "gelu", "elu", "tanh", "sigmoid", "leaky_relu" ).
        """
        super(CustomSoftmax, self).__init__()
        self.generator = torch.Generator().manual_seed(seed)
        assert(initialization in ["xavier uniform" , " xavier normal" ,"kaiming uniform" ])
        self.initialization=initialization
        layers = []

        # Choose the activation function
        activations = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "elu": nn.ELU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
            "leaky_relu": nn.LeakyReLU
        }
        if activation not in activations:
            raise ValueError(f"Unsupported activation function '{activation}'. Choose from {list(activations.keys())}.")
        activation_fn = activations[activation]
        self.activation=activation

        # Case 1: 0 Hidden Layers
        if not hidden_layers:
            layers.append(nn.Linear(input_dim, output_dim))

        # Case 2: 1 or More Hidden Layers
        else:
            # Input layer
            layers.append(nn.Linear(input_dim, hidden_layers[0]))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_layers[0]))
            layers.append(activation_fn())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            # Hidden layers
            for i in range(1, len(hidden_layers)):
                layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
                if use_batch_norm:
                    layers.append(nn.BatchNorm1d(hidden_layers[i]))
                layers.append(activation_fn())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

            # Output layer (no activation or batch normalization)
            layers.append(nn.Linear(hidden_layers[-1], output_dim))
            layers.append(nn.Softmax(dim=1))

        self.network = nn.Sequential(*layers)
        #self.initialize()  not called because I am not sure it is properly done

    def initialize(self):
        for l in range(len(self.network)):
            custom_init(self.network[l], generator=self.generator , init_type=self.initialization, nonlinearity=self.activation)

    def forward(self, x):
        return self.network(x)
