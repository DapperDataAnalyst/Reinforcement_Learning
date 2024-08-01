import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from algo import ValueFunctionWithApproximation

class ValueFunctionWithNN(ValueFunctionWithApproximation):
    def __init__(self, state_dims):
        """
        state_dims: the number of dimensions of state space
        """
        super(ValueFunctionWithNN, self).__init__()
        
        # Define the neural network
        self.model = nn.Sequential(
            nn.Linear(state_dims, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Output layer
        )
        
        # Define the optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.999))
        
        # Define the loss function
        self.loss_fn = nn.MSELoss()

    def __call__(self, s):
        self.model.eval()  # Set model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation
            s_tensor = torch.tensor(s, dtype=torch.float32)
            value = self.model(s_tensor).item()
        return value

    def update(self, alpha, G, s_tau):
        self.model.train()  # Set model to training mode
        
        s_tau_tensor = torch.tensor(s_tau, dtype=torch.float32)
        G_tensor = torch.tensor(G, dtype=torch.float32)
        
        # Zero the parameter gradients
        self.optimizer.zero_grad()
        
        # Forward pass
        predicted_value = self.model(s_tau_tensor)
        
        # Compute the loss
        loss = self.loss_fn(predicted_value, G_tensor)
        
        # Backward pass
        loss.backward()
        
        # Update the weights
        self.optimizer.step()
