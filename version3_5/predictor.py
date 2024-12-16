import torch
import training
import neural_network
import dev_load_data as load_data

##########
train=True
calculate=True
use_best_model=True
##########

checkpoint_path="model_state.pth"
best_model_path="best_model.pth"
model_train=training.trainer()

data_loader=load_data.give_data()
data=data_loader.get_data(eval=True)

if train:
    model_train.train(20)

cnn_input_channels = 8          # e.g., grayscale image
cnn_output_channels = 64        # starting CNN output channels
lstm_input_size = 128            # flattened CNN output channels
lstm_hidden_size = 256          # LSTM hidden size
transformer_input_size = 32     # transformer input size (d_model)
transformer_num_heads = 8       # transformer number of heads
transformer_hidden_dim = 512    # transformer hidden dimension (feedforward layer)

model = neural_network.AdvancedHybridModel(
            cnn_input_channels=cnn_input_channels,
            cnn_output_channels=cnn_output_channels,
            lstm_input_size=lstm_input_size,
            lstm_hidden_size=lstm_hidden_size,
            transformer_input_size=transformer_input_size,
            transformer_num_heads=transformer_num_heads,
            transformer_hidden_dim=transformer_hidden_dim,
        )
if use_best_model:
    model.load_state_dict(torch.load(best_model_path))
else:
    model.load_state_dict(torch.load(checkpoint_path))
   

model.eval()
with torch.no_grad():
    output=model(data)-5

print(output)
if calculate:
    original=float(input("Enter close price: "))
    print(original+(original*output/100))