import torch
import torch.nn as nn

class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, num_factors=30):
        super().__init__
        self.user_factors = nn.Embedding(num_users, num_factors)
        self.item_factors = nn.Embedding(num_items, num_factors)
        self.user_bias = nn.Embedding(num_items, 1)
        self.item_bias = nn.Embedding(num_items, 1)

        nn.init.normal_(self.user_factors.weight, std=0.01)
        nn.init.normal_(self.item_factors.weight, std=0.01)
        nn.init.zeros_(self.item_bias.weight)
        nn.init.zeros_(self.item_factors.weight)
    def forward(self, user_id, item_id):
        P_u = self.user_factors(user_id)  # User latent vector
        Q_i = self.item_factors(item_id)  # Item latent vector
        b_u = self.user_bias(user_id).squeeze()  # User bias
        b_i = self.item_bias(item_id).squeeze()  # Item bias
        outputs = (P_u * Q_i).sum(dim=1) + b_u + b_i
        return outputs
    def rmse(preds, targets):
        return torch.sqrt(torch.mean((preds-targets)**2))
    def evaluate(model, test_load, device)