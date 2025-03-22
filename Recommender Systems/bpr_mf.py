import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

class MF(nn.Module):
    """
    Matrix Factorization model for recommendation systems.

    Parameters:
        num_users (int): Number of users.
        num_items (int): Number of items.
        num_factors (int): Number of latent factors (K) in user-item decomposition.
        device (torch.device): The device (CPU/GPU) used for computation.

    Attributes:
        user_factors (nn.Embedding): Latent factors for users.
        item_factors (nn.Embedding): Latent factors for items.
        user_bias (nn.Embedding): Bias terms for users.
        item_bias (nn.Embedding): Bias terms for items.
    """

    def __init__(self, num_users, num_items, num_factors, device):
        super().__init__()
        self.device = device

        # User and item embeddings (latent factors)
        self.user_factors = nn.Embedding(num_users, num_factors)
        self.item_factors = nn.Embedding(num_items, num_factors)

        # Bias terms for users and items
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)

        # Parameter initialization
        nn.init.normal_(self.user_factors.weight, std=0.01)
        nn.init.normal_(self.item_factors.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

        # Move model to the specified device
        self.to(device)

    def forward(self, user_id, item_id):
        """
        Forward pass to compute predicted scores.

        Parameters:
            user_id (torch.Tensor): Tensor containing user indices.
            item_id (torch.Tensor): Tensor containing item indices.

        Returns:
            torch.Tensor: Predicted scores computed as:
                prediction = (P * Q).sum(dim=1) + b_u + b_i
        """
        user_id = user_id.to(self.device)
        item_id = item_id.to(self.device)
        P_u = self.user_factors(user_id)  # User latent vectors
        Q_i = self.item_factors(item_id)  # Item latent vectors
        b_u = self.user_bias(user_id).squeeze()  # User bias
        b_i = self.item_bias(item_id).squeeze()  # Item bias

        return (P_u * Q_i).sum(dim=1) + b_u + b_i

    def evaluate_rmse(self, test_loader):
        """
        Evaluates the model's performance using RMSE on a test or validation set (for MSE-based evaluations).

        Parameters:
            test_loader (DataLoader): DataLoader containing the test/validation data.

        Returns:
            float: RMSE of the model’s predictions.
        """
        self.eval()  # Set model to evaluation mode
        criterion = nn.MSELoss()
        total_loss, count = 0, 0

        with torch.no_grad():
            for users, items, ratings in test_loader:
                users = users.to(self.device)
                items = items.to(self.device)
                ratings = ratings.float().to(self.device)
                preds = self.forward(users, items)
                loss = criterion(preds, ratings)
                total_loss += loss.item() * len(ratings)
                count += len(ratings)

        return np.sqrt(total_loss / count)

    def evaluate_bpr(self, val_loader, reg=0):
        """
        Evaluates the model using the BPR loss over a validation set containing triplets.

        Parameters:
            val_loader (DataLoader): DataLoader containing (users, pos_items, neg_items).
            reg (float): Regularization coefficient (optional).

        Returns:
            float: Average BPR loss on the validation set.
        """
        self.eval()
        total_loss, count = 0, 0

        with torch.no_grad():
            for users, pos_items, neg_items in val_loader:
                users    = users.to(self.device)
                pos_items = pos_items.to(self.device)
                neg_items = neg_items.to(self.device)

                # Compute scores for positive and negative items
                score_pos = self.forward(users, pos_items)
                score_neg = self.forward(users, neg_items)

                # BPR loss for the batch
                loss = -torch.log(torch.sigmoid(score_pos - score_neg)).sum()

                # L2 Regularization (optional)
                l2_reg = torch.norm(self.user_factors.weight, p=2) + torch.norm(self.item_factors.weight, p=2)
                loss += reg * l2_reg

                total_loss += loss.item()
                count += len(users)

        return total_loss / count

    def train_model_bpr(self, train_loader, val_loader, num_epochs, num_step=10, lr=0.002, reg=1e-5, gamma=0.1):
        """
        Trains the Matrix Factorization model using the BPR loss function.

        Parameters:
            train_loader (DataLoader): DataLoader for training data (triplets: users, pos_items, neg_items).
            val_loader (DataLoader): DataLoader for validation data (triplets).
            num_epochs (int): Number of training epochs.
            num_step (int, optional): Step size for learning rate decay. Default is 10.
            lr (float, optional): Learning rate. Default is 0.002.
            reg (float, optional): Regularization strength (L2). Default is 1e-5.
            gamma (float, optional): Decay factor for learning rate. Default is 0.1.

        Returns:
            tuple: (train_losses, val_losses), lists containing training and validation losses per epoch.
        """
        optimizer = optim.Adam(self.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=num_step, gamma=gamma)
        train_losses, val_losses = [], []

        for epoch in range(num_epochs):
            self.train()  # Set model to training mode
            total_loss, count = 0, 0
            start_time = time.time()

            for users, pos_items, neg_items in train_loader:
                users    = users.to(self.device)
                pos_items = pos_items.to(self.device)
                neg_items = neg_items.to(self.device)

                optimizer.zero_grad()
                # Get scores for positive and negative items
                score_pos = self.forward(users, pos_items)
                score_neg = self.forward(users, neg_items)

                # Compute the BPR loss
                loss = -torch.log(torch.sigmoid(score_pos - score_neg)).sum()

                # L2 Regularization (optional)
                l2_reg = torch.norm(self.user_factors.weight, p=2) + torch.norm(self.item_factors.weight, p=2)
                loss += reg * l2_reg

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                count += len(users)

            # Average training loss for the epoch
            train_loss = total_loss / count
            # Evaluate on the validation set using BPR loss
            val_loss = self.evaluate_bpr(val_loader, reg=reg)

            scheduler.step()
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            elapsed_time = time.time() - start_time
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {elapsed_time:.2f}s")

        return train_losses, val_losses
