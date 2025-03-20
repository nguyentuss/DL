import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from utils import transform2d_ml

class AutoRec(nn.Module):
    """
    AutoRec is a collaborative filtering-based recommender system 
    that uses autoencoders (a type of neural network) 
    for dimensionality reduction. It is designed to 
    predict user ratings or preferences for items
    (e.g., movies, products) based on observed ratings.

    Parameters:
        input_dim: Number of items or users (depends on whether it's U-AutoRec or I-AutoRec)
        hidden_dim: Dimensionality of the hidden layer (latent representation)
        device (torch.device): The device (CPU/GPU) used for computation
    
    Reference:
        https://users.cecs.anu.edu.au/~akmenon/papers/autorec/autorec-paper.pdf
    """

    def __init__(self, input_dim, hidden_dim, device):
        super().__init__()
        
        # Autorencoder architecture
        self.encode = nn.Linear(input_dim, hidden_dim)
        self.decode = nn.Linear(hidden_dim, input_dim)
        
        # Set device
        self.device = device

        # Move model to the specified device
        self.to(device)
    
    def forward(self, x):
        """
        Forward pass through the encoder and decoder

        Parameters:
            x (torch.Tensor): Input Tensor (user-item or item-user matrix).

        Returns:
            outputs (torch.Tensor): Reconstructed input tensor (predicted the user-item interactions)
        """
        x = x.to(self.device)  # Move input tensor to the model's device

        hidden = torch.sigmoid(self.encode(x))
        outputs = torch.sigmoid(self.decode(hidden))

        return outputs
    
    def evaluate(self, test_loader):
        """
        Evaluates the model's performance using RMSE on a test or validation set.

        Parameters:
            test_loader (DataLoader): DataLoader containing the test/validation data.

        Returns:
            float: RMSE of the model’s predictions.
        """
        self.eval() # Set model to evaluation mode
        criterion = nn.MSELoss()
        total_loss, count = 0, 0

        with torch.no_grad():
            for batch in test_loader:
                inputs = batch[0].to(self.device)  # Move inputs to device
                #Compute the forward NN        
                preds= self.forward(inputs)
                loss = criterion(preds, inputs)
                print(f"Predictions: {preds[:5]}")
                print(f"True inputs: {inputs[:5]}")
                total_loss += loss
                count += 1
        return np.sqrt((total_loss / count).cpu().numpy())  # Move to CPU and convert


    def train_model(self, train_loader, val_loader, num_epochs, num_step=10, lr=0.002, reg=1e-5, gamma=0.1):
        """
        Trains the AutoRec model using Adam optimizer and MSE loss.

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
            self.train()
            total_loss, count = 0, 0
            start_time = time.time()

            for batch in train_loader:
                inputs = batch[0].to(self.device)  # Move batch to the correct device

                # Zero the parameter gradient
                optimizer.zero_grad()

                #Compute the forward NN        
                preds= self.forward(inputs)
                loss = loss_fn(preds, inputs)

                # L2 Regularization
                l2_reg = sum(torch.norm(param, p=2) for param in self.parameters())
                loss += reg * l2_reg
                
                loss.backward()
                optimizer.step()

                total_loss += loss.item() 
                count += 1
            # Compute training and validation loss
            train_loss = total_loss / count
            # val_loss = self.evaluate(val_loader)
            val_loss = 0 

            # Update learning rate
            scheduler.step()

            # Store losses
            train_losses.append(train_loss)
            # val_losses.append(val_loss)
            elapsed_time = time.time() - start_time
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {elapsed_time:.2f}s")
        return train_losses, val_losses


