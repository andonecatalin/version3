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
        global_pool = F.adaptive_avg_pool1d(conv_out, 1)
        global_pool = global_pool.permute(1,0)  # Reshape to [batch, channels]
        fc_out = self.fc2(F.relu(self.fc1(global_pool)))
        attention = self.softmax(fc_out)  # Reshape to [batch, channels, 1]
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
    """Conformer Layer for capturing both local and global context"""
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
    

class AttentionCombine(nn.Module):
    def __init__(self, input_shapes, output_dim):
        super(AttentionCombine, self).__init__()
        self.projections = nn.ModuleList(
            [nn.Linear(input_shape, output_dim) for input_shape in input_shapes]
        )

    def forward(self, outputs):
        # Transform all outputs to the same size
        projected = [proj(output) for proj, output in zip(self.projections, outputs)]
        stacked = torch.stack(projected, dim=1)  # [batch_size, num_outputs, output_dim]
        attention_weights = torch.softmax(torch.sum(stacked, dim=-1), dim=0).unsqueeze(-1)
        combined = (attention_weights * stacked).sum(dim=1)  # Weighted sum
        return combined



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
        self.fc = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),           
            nn.Linear(512, 1024),  
            nn.ReLU(),           
            nn.Linear(1024, 1),  
        )
        self.attentionCombine=AttentionCombine(input_shapes=[1024,4096,1024],output_dim=256)   

    def forward(self, x):
        # Pass through CNN with Dynamic Convolutional Attention
        #print(x.shape)
        x_cnn = self.dynamic_conv_attention(x)
        x_cnn = x_cnn.permute(1,0)  # Reshape to [batch, seq_len, features]

        # Pass through LSTM Layer with Layer Normalization
        lstm_out, _ = self.lstm(x_cnn)
        lstm_out = self.lstm_layernorm(lstm_out)

        # Transformer Encoder with Positional Encoding and Conformer Layer
        transformer_in = self.positional_encoding(x[-1])  # Prepare for Transformer [seq_len, batch, dim]
        transformer_out = self.conformer(transformer_in)  # Transformer expects [seq_len, batch, dim]
        transformer_out = transformer_out.permute(1, 0, 2)  # Reshape back to [batch, seq_len, dim]
        transformer_out = transformer_out.flatten()  # Take the last time step

        x_cnn=x_cnn.flatten()
        lstm_out=lstm_out.flatten()
        result=self.attentionCombine([x_cnn,lstm_out,transformer_out])

        output = self.fc(result)
        return output
