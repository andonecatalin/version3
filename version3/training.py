import torch
import torch.nn as nn
import seaborn as sns
import load_data

import sys
import neural_network


class ModifiedHuberLoss(nn.Module):
    def __init__(self, delta=2.5, alpha=0.2):
        """
        delta (float): Threshold where loss transitions from quadratic to linear.
        alpha (float): Scaling factor for the linear penalty in the large error region.
        """
        super(ModifiedHuberLoss, self).__init__()
        self.delta = delta
        self.alpha = alpha

    def forward(self, y_true, y_pred):
        error = y_true - y_pred
        abs_error = torch.abs(error)
        
        # Quadratic loss for small errors
        small_error_loss = 0.5 * error**2
        
        # Reduced linear penalty for large errors
        large_error_loss = self.alpha * self.delta * abs_error - 0.5 * self.delta**2
        
        # Apply condition for transitioning between quadratic and linear regions
        loss = torch.where(abs_error <= self.delta, small_error_loss, large_error_loss)
        
        return loss.mean()
    
    

class trainer:
    def __init__(self):
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Pythorch is using {device}')


        self.cnn_input_channels = 5          # e.g., grayscale image
        self.cnn_output_channels = 32        # starting CNN output channels
        self.lstm_input_size = 64            # flattened CNN output channels
        self.lstm_hidden_size = 128          # LSTM hidden size
        self.transformer_input_size = 32     # transformer input size (d_model)
        self.transformer_num_heads = 8       # transformer number of heads
        self.transformer_hidden_dim = 256     # transformer hidden dimension (feedforward layer)

    def train(self,epochs=100):
        # Instantiate the model
        model = neural_network.AdvancedHybridModel(
            cnn_input_channels=self.cnn_input_channels,
            cnn_output_channels=self.cnn_output_channels,
            lstm_input_size=self.lstm_input_size,
            lstm_hidden_size=self.lstm_hidden_size,
            transformer_input_size=self.transformer_input_size,
            transformer_num_heads=self.transformer_num_heads,
            transformer_hidden_dim=self.transformer_hidden_dim,
        )

        loader=load_data.give_data()

        train_loader=loader.get_data()

        type(train_loader)

        losses=[]
        epoch_losses=[]

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-2)
        criterion=ModifiedHuberLoss()

        for epoch in range(epochs):
            model.train()
            epoch_loss=0.0

            for input, label in train_loader:
                
                input=input.squeeze(0)
                optimizer.zero_grad()
                output=model(input)
                loss=criterion(output,label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss+=loss.item()
            
            avg_loss=epoch_loss/len(train_loader)
            epoch_losses.append(epoch_loss)
            losses.append(avg_loss)
            print(f'\rEpoch[{epoch+1}/{epochs}], Loss: {avg_loss:.4f}', end="")
            sys.stdout.flush()
        torch.save(model.state_dict(), 'model_state.pth')
        sns.displot(losses)
        sns.lineplot(losses)
        
