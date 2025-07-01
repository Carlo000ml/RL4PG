import torch.nn as nn
from ...DeepL.Models import CustomRegressor
class DQNetwork:
    def __new__(cls, input_dim, output_dim, hidden_layers=None, dropout=0.0, use_batch_norm=False, activation="leaky_relu"):
        return CustomRegressor(input_dim, output_dim, hidden_layers, dropout, use_batch_norm, activation)


class DuelingQNetwork(nn.Module):
    """
    Implements a Dueling Q-Network.

    The network consists of:
    - A shared feature extractor (`feature_extractor`).
    - A value stream (`value_stream`) that estimates V(s) (the value of being in a state).
    - An advantage stream (`advantage_stream`) that estimates A(s, a) (the relative value of each action).
    - A normalization method (`ad_norm`) to compute the final Q-values.

    Args:
        input_dim (int): Number of input features (state representation size).
        output_dim (int): Number of possible actions (Q-value outputs).
        hidden_layers_shared_structure (list, optional): Hidden layer sizes for the shared feature extractor. The output dim is equal to the first dimension of the two streams
        structure_value_stream (list): List of layer sizes for the value stream. The output is fixed to 1.
        structure_advantage_stream (list): List of layer sizes for the advantage stream. 
        dropout (float, optional): Dropout rate for regularization.
        use_batch_norm (bool, optional): Whether to apply batch normalization.
        activation (str, optional): Activation function to use (e.g., "relu" or "tanh").
        advantage_comparison (str, optional): Method for normalizing advantage values. Options: ["mean", "do nothing"].
    """
    def __init__(self, input_dim, output_dim,hidden_layers_shared_structure=None,structure_value_stream=None,structure_advantage_stream=None,dropout=0.0,use_batch_norm=False,activation="leaky_relu",advantage_comparison="mean"):
        super(DuelingQNetwork, self).__init__()
        assert(structure_value_stream[0]==structure_advantage_stream[0]), "First dimension of value stream and advantage stream must be equal"
        assert(advantage_comparison in ["mean" , "max"]) , "advantage_comparison must be either 'mean' or 'do nothing'"

        self.feature_extractor=CustomRegressor(input_dim=input_dim, output_dim=structure_value_stream[0],hidden_layers=hidden_layers_shared_structure,dropout=dropout, use_batch_norm=use_batch_norm,activation=activation)
        self.value_stream=CustomRegressor(input_dim=structure_value_stream[0],output_dim=1,hidden_layers=structure_value_stream[1:], dropout=dropout, use_batch_norm=use_batch_norm, activation=activation)
        self.advantage_stream=CustomRegressor(input_dim=structure_advantage_stream[0] , output_dim=output_dim , hidden_layers=structure_advantage_stream[1:] , dropout=dropout, use_batch_norm=use_batch_norm, activation=activation)
        self.ad_norm=advantage_comparison



    def forward(self, x):
        features = self.feature_extractor(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        # Compute Q-values: Q(s, a) = V(s) + A(s, a) - mean(A(s, a'))
        if self.ad_norm == "mean":
            # Subtract the mean of advantages across the action dimension.
            advantage_offset = advantage.mean(dim=1, keepdim=True)  # shape: (batch_size, 1)
            q_values = value + (advantage - advantage_offset)
        else:
            # Subtract the max of advantages across the action dimension.
            advantage_offset = advantage.max(dim=1, keepdim=True)[0]  # shape: (batch_size, 1)
            q_values = value + (advantage - advantage_offset)


        return q_values
    