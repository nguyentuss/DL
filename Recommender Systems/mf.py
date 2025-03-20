import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from torch.utils.data import DataLoader, TensorDataset

class MF(nn.Module):
    """
    Matrix factorization for recommended system
    
    ---

    Parameters:
        - num_users : number of user
        - num_items : number of items
        - num_factors : number of hidden parameter (N x M (user item) = N x K (user latent) * K x M (item latent))
        - device : which GPU or CPU we will use
    """
    def __init__(self, num_users, num_items, num_factors, device):
        super().__init__()

        self.device = device
        self.user_factors = nn.Embedding(num_users, num_factors) # adding layers
        self.item_factors = nn.Embedding(num_items, num_factors) # adding layers
        self.user_bias = nn.Embedding(num_items, 1)
        self.item_bias = nn.Embedding(num_items, 1)

        nn.init.normal_(self.user_factors.weight, std=0.01)
        nn.init.normal_(self.item_factors.weight, std=0.01)
        nn.init.zeros_(self.item_bias.weight)
        nn.init.zeros_(self.item_factors.weight)
        self.to(device)
    
    def _forward(self, user_id, item_id):
        P_u = self.user_factors(user_id)  # User latent vector
        Q_i = self.item_factors(item_id)  # Item latent vector
        b_u = self.user_bias(user_id).squeeze()  # User bias
        b_i = self.item_bias(item_id).squeeze()  # Item bias
        outputs = (P_u * Q_i).sum(dim=1) + b_u + b_i 
        return outputs
    
    def _evaluator(self, test_loader):
        self.eval() # Set to the evalute mode (no training)
        criterion = nn.MSELoss()
        total_loss, count = 0, 0
        with torch.no_grad():
            for users, items, ratings in test_loader:
                users, items, ratings = users.to(self.device), items.to(self.device), ratings.float().to(self.device)
                preds = self._forward(users, items)
                loss = criterion(preds, ratings.float())
                total_loss += loss.item() * len(ratings)
                count += len(ratings)
        return np.sqrt(total_loss / count)
    def _train(self, train_loader, val_loader, num_epochs, num_step = 10, lr = 0.002, reg=1e-5, gamma=0.1):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        
        # After num_step epochs, the learning rate will decay a value "lr"
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=num_step, gamma=gamma)
        
        loss_fn = nn.MSELoss()
        train_losses, val_losses = [], []
        for epoch in range(num_epochs):
            self.train() # Set to training mode
            total_loss, count = 0, 0
            start_time = time.time()
            for users, items, ratings in train_loader:
                users, items, ratings = users.to(self.device), items.to(self.device), ratings.float().to(self.device)
                optimizer.zero_grad() # reset the grad in optimizer
                preds = self._forward(users, items)
                loss = loss_fn(preds, ratings)
                
                # Regularization
                l2_reg = torch.norm(self.item_factors.weight, p = 2) + torch.norm(self.user_factors.weight, p = 2)
                loss += reg * l2_reg

                loss.backward() # backprop
                optimizer.step() # update parameter 
                # print(loss.item())
                total_loss += loss.item() * len(ratings) 
                count += len(ratings)
            train_loss = total_loss / count 
            # print(train_loss)
            val_loss = self._evaluator(val_loader)

            scheduler.step()
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            elapsed_time = time.time() - start_time
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {elapsed_time:.2f}s")
        return train_losses, val_losses


                