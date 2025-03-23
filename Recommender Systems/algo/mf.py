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
        Forward pass to compute predicted ratings.

        Parameters:
            user_id (torch.Tensor): Tensor containing user indices.
            item_id (torch.Tensor): Tensor containing item indices.

        Returns:
            torch.Tensor: Predicted ratings using the formula:
                prediction = (P * Q).sum(dim=1) + b_u + b_i
        """
        user_id = user_id.to(self.device)
        item_id = item_id.to(self.device)
        P_u = self.user_factors(user_id)  # User latent vectors
        Q_i = self.item_factors(item_id)  # Item latent vectors
        b_u = self.user_bias(user_id).squeeze()  # User bias
        b_i = self.item_bias(item_id).squeeze()  # Item bias

        return (P_u * Q_i).sum(dim=1) + b_u + b_i

    def evaluate(self, test_loader):
        """
        Evaluates the model's performance using RMSE on a test or validation set.

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
                users, items, ratings = users.to(self.device), items.to(self.device), ratings.float().to(self.device)
                preds = self.forward(users, items)
                loss = criterion(preds, ratings)
                total_loss += loss.item() * len(ratings)
                count += len(ratings)

        return np.sqrt(total_loss / count)  # Compute RMSE

    def train_model(self, train_loader, val_loader, num_epochs, num_step=10, lr=0.002, reg=1e-5, gamma=0.1):
        """
        Trains the Matrix Factorization model using Adam optimizer and MSE loss.

        Parameters:
            train_loader (DataLoader): DataLoader for training set.
            val_loader (DataLoader): DataLoader for validation set.
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
        loss_fn = nn.MSELoss()

        train_losses, val_losses = [], []

        for epoch in range(num_epochs):
            self.train()  # Set model to training mode
            total_loss, count = 0, 0
            start_time = time.time()

            for users, items, ratings in train_loader:
                users, items, ratings = users.to(self.device), items.to(self.device), ratings.float().to(self.device)

                optimizer.zero_grad()  # Reset gradients
                preds = self.forward(users, items)
                loss = loss_fn(preds, ratings)

                # L2 Regularization
                l2_reg = torch.norm(self.user_factors.weight, p=2) + torch.norm(self.item_factors.weight, p=2)
                loss += reg * l2_reg

                loss.backward()  # Backpropagation
                optimizer.step()  # Update parameters

                total_loss += loss.item() * len(ratings)
                count += len(ratings)

            # Compute training and validation loss
            train_loss = total_loss / count
            val_loss = self.evaluate(val_loader)

            # Update learning rate
            scheduler.step()

            # Store losses
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            elapsed_time = time.time() - start_time
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Time: {elapsed_time:.2f}s")

        return train_losses, val_losses
