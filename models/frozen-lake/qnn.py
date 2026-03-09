import torch
from torch import nn
import torch.nn.functional as F

class QNN(nn.Module):
    """
    Neural Net over Q-Table
    """
    def __init__(self, n_layers=3, input_dim=4, hidden_dim=10, output_dim=4):
        
        super().__init__()

        self.state_embedder = nn.Embedding(num_embeddings=64, embedding_dim=4)

        mlp_layers = []
        
        in_features = input_dim
        for _ in range(n_layers):
            mlp_layers.append(nn.Linear(in_features, hidden_dim))
            mlp_layers.append(nn.GELU())
            in_features=hidden_dim
        mlp_layers.append(nn.Linear(hidden_dim, output_dim))
        self.ff_layers = nn.Sequential(*mlp_layers)


    def forward(self, x):
        # Embed our actions 
        embedded_state = self.state_embedder(x)

        # Non-Linear MLP
        ff_layers = self.ff_layers(embedded_state)

        # Softmax to get the probs
        # action_value = self.softmax(ff_layers)
        return ff_layers



    
if __name__ == "__main__":

    index = torch.tensor([5])  # Just the index, shape [1]
    model = QNN(n_layers=5)
    output = model(index)  # Embedding handles the lookup, outputs [1, 4]
    