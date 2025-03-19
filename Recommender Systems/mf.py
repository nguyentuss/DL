import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from torch.utils.data import DataLoader, TensorDataset

class MF(nn.Module):
    def __init__(self, num_users, num_items, num_factors=30, device):
        super(MF, self).__init__()

        self.device = device
        self.user_factors = nn.Embedding(num_users, num_factors)
        self.item_factors = nn.Embedding(num_items, num_factors)
        self.user_bias = nn.Embedding(num_items, 1).squeeze()
        self.item_bias = nn.Embedding(num_items, 1).squeeze()

        nn.init.normal_(self.user_factors.weight, std=0.01)
        nn.init.normal_(self.item_factors.weight, std=0.01)
        nn.init.zeros_(self.item_bias.weight)
        nn.init.zeros_(self.item_factors.weight)
        self.to(device)
    
    def forward(self, user_id, item_id):
        P_u = self.user_factors(user_id)  # User latent vector
        Q_i = self.item_factors(item_id)  # Item latent vector
        b_u = self.user_bias(user_id).squeeze()  # User bias
        b_i = self.item_bias(item_id).squeeze()  # Item bias
        outputs = (P_u * Q_i).sum(dim=1) + b_u + b_i
        return outputs
    
    def evaluator(self, test_loader):
        self.eval()
        criterion = nn.MSELoss()
        with torch.no_grad():
            for users, items, ratings in test_loader:
                users, items, ratings = users.to(self.device), items.to(self.device), ratings.float().to(self.device)
                preds = self(users, items)
                loss = criterion(preds, ratings.float())
                total_loss += loss.item() * len(ratings)
                