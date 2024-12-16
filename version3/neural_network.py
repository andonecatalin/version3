import torch.nn as nn
import torch.nn.functional as F 
import torch
import numpy as np 



class DynamicConvAttention(nn.Module):
    """Dynamic Convolutional Attention Block"""
    def __init__(self, in_channels, out_channels, kernel_size=3, reduction_ratio=2):
        super(DynamicConvAttention, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=1)
        self.fc1 = nn.Linear(out_channels, out_channels // reduction_ratio)
        self.fc2 = nn.Linear(out_channels // reduction_ratio, out_channels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        conv_out = self.conv(x)
        global_pool = F.adaptive_avg_pool1d(conv_out,1)
        global_pool=global_pool.permute(1,0)
        fc_out = self.fc2(F.relu(self.fc1(global_pool)))
        attention = self.softmax(fc_out)
        attention=attention.permute(1,0)
        return conv_out * attention


class PositionalEncoding(nn.Module):
    """Positional Encoding for Transformer"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x
    

class ConformerLayer(nn.Module):
    #Conformer Layer for capturing both local and global context
    def __init__(self, d_model, num_heads, conv_hidden_dim, dropout=0.1):
        super(ConformerLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.conv_module = nn.Sequential(
            nn.Conv1d(d_model, conv_hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(conv_hidden_dim, d_model, kernel_size=1)
        )
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Self-Attention Block
        attn_out, _ = self.self_attn(x, x, x)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # Convolution Block
        x_conv = self.conv_module(x.permute(1, 2, 0)).permute(2, 0, 1)
        x = x + self.dropout(x_conv)
        x = self.norm2(x)
        return x
    


class AdvancedHybridModel(nn.Module):
    def __init__(self, cnn_input_channels, cnn_output_channels, lstm_input_size, lstm_hidden_size, 
                 transformer_input_size, transformer_num_heads, transformer_hidden_dim):
        super(AdvancedHybridModel, self).__init__()

        # Dynamic Convolutional Attention Block for enhanced spatial feature extraction
        self.dynamic_conv_attention = nn.Sequential(
            DynamicConvAttention(cnn_input_channels, cnn_output_channels),
            nn.MaxPool1d(2),
            nn.Dropout(0.33),
            DynamicConvAttention(cnn_output_channels, cnn_output_channels * 2),
            nn.MaxPool1d(2),
            nn.Dropout(0.33)
        )
        
        # Bidirectional LSTM with LayerNorm
        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=lstm_hidden_size, num_layers=2, 
                            batch_first=True, bidirectional=True)
        self.lstm_layernorm = nn.LayerNorm(lstm_hidden_size * 2)
        
        # Transformer Encoder with Conformer Layer and Positional Encoding
        self.positional_encoding = PositionalEncoding(d_model=transformer_input_size)
        self.conformer = ConformerLayer(d_model=transformer_input_size, num_heads=transformer_num_heads, 
                                        conv_hidden_dim=transformer_hidden_dim)


        # Classifier
        self.fc=nn.Sequential(
            nn.Linear(48,64),
            nn.ReLU(),           
            nn.Linear(64, 128),  
            nn.ReLU(),           
            nn.Linear(128, 1),  
        )

    def forward(self, x):
        # Pass through CNN with Dynamic Convolutional Attention
        transformer_in=x
        x_cnn = self.dynamic_conv_attention(x)

        x_cnn=x_cnn.permute(1,0)
        # Pass through LSTM Layer with Layer Normalization
        lstm_out, _ = self.lstm(x_cnn)
        lstm_out = self.lstm_layernorm(lstm_out)

        # Transformer Encoder with Positional Encoding and Conformer Layer
        transformer_in = self.positional_encoding(x[4:,:])
        transformer_out = self.conformer(transformer_in.permute(1, 0, 2))  # Transformer expects [seq_len, batch, dim]
        transformer_out = transformer_out.permute(1, 0, 2)  # Reshape back to [batch, seq_len, dim]
        transformer_out=transformer_out.squeeze()

        result=torch.cat((x_cnn[:,-1],lstm_out[:,-1],transformer_out),dim=0)
        # Final Classification Layer
        output = self.fc(result)
        return output