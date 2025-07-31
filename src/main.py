import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

device = "cuda:0" if torch.cuda.is_available() else "cpu"
conv_filters = [[1,32],[3,32],[5,64],[7,128]]
embedding_size = output_dim = 256
d_ff = 256
n_heads = 8
d_k = 16
n_layer = 1

smi_vocab_size = 53
seq_vocab_size = 21

seed = 990721

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.manual_seed(seed)
np.random.seed(seed)

class Squeeze(nn.Module):
    def forward(self, input: torch.Tensor):
        return input.squeeze()

def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], True is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]

class ConvEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_size, conv_filters, output_dim, type):
        super().__init__()
        if type == 'seq':
            self.embed = nn.Embedding(vocab_size, embedding_size)
        elif type == 'poc':
            self.embed = nn.Embedding(vocab_size, embedding_size, padding_idx=0)

        self.convolutions = nn.ModuleList()
        for kernel_size, out_channels in conv_filters:
            conv = nn.Conv1d(embedding_size, out_channels, kernel_size, padding = (kernel_size - 1) // 2)
            self.convolutions.append(conv)
        self.num_filters = sum([f[1] for f in conv_filters])
        self.projection = nn.Linear(self.num_filters, output_dim)

    def forward(self, inputs):
        embeds = self.embed(inputs).transpose(-1,-2)  # (batch_size, embedding_size, seq_len)
        conv_hidden = []
        for layer in self.convolutions:
            conv = F.relu(layer(embeds)) 
            conv_hidden.append(conv)
        res_embed = torch.cat(conv_hidden, dim = 1).transpose(-1,-2)  # (batch_size, seq_len, num_filters)
        res_embed = self.projection(res_embed) 
        return res_embed

# ResNet-style residual blocks
class ResNetBlock(nn.Module):
    def __init__(self, input_dim, activation=F.relu):
        super().__init__()
        self.linear = nn.Linear(input_dim, input_dim)
        self.activation = activation
        self.layer_norm = nn.LayerNorm(input_dim)  # Optional: Add LayerNorm for stability

    def forward(self, x):
        residual = x
        out = self.linear(x)
        out = self.activation(out)
        return self.layer_norm(residual + out)  # Skip connection + LayerNorm

class ResNet(nn.Module):
    def __init__(self, input_dim, num_layers, activation=F.relu):
        super().__init__()
        self.input_dim = input_dim
        self.layers = nn.ModuleList([ResNetBlock(input_dim, activation) for _ in range(num_layers)])

    def forward(self, inputs):
        curr_inputs = inputs
        for layer in self.layers:
            curr_inputs = layer(curr_inputs)
        return curr_inputs

class SelfAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_Q = nn.Linear(embedding_size, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(embedding_size, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(embedding_size, d_k * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_k, embedding_size, bias=False)
        self.ln = nn.LayerNorm(embedding_size)
        self.attn = SelfAttention()

    def forward(self, input_Q, input_K, input_V, attn_mask):
        batch_size = input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1,2)
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_k).transpose(1,2)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        context = self.attn(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_k)
        output = self.fc(context)
        return self.ln(output)

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embedding_size, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, embedding_size, bias=False)
        )
        self.ln = nn.LayerNorm(embedding_size)

    def forward(self, inputs):
        residual = inputs
        output = self.fc(inputs)
        return self.ln(output + residual) 

class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.multi = MultiHeadAttention()
        self.feed = FeedForward()

    def forward(self, en_input, attn_mask):
        context = self.multi(en_input, en_input, en_input, attn_mask)
        output = self.feed(context + en_input)  # Residual connection
        return output

class Seq_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq_emb = ConvEmbedding(seq_vocab_size, embedding_size, conv_filters, output_dim, 'seq')
        self.poc_emb = ConvEmbedding(seq_vocab_size, embedding_size, conv_filters, output_dim, 'poc')
        self.resnet = ResNet(embedding_size, n_layer)  # Replaced Highway with ResNet
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layer)])

    def forward(self, seq_input, poc_input):
        output_emb = self.seq_emb(seq_input) + self.poc_emb(poc_input)
        output_emb = self.resnet(output_emb)  # Pass through ResNet blocks
        enc_self_attn_mask = get_attn_pad_mask(seq_input, seq_input)
        for layer in self.layers:
            output_emb = layer(output_emb, enc_self_attn_mask)
        return output_emb

class Smi_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = ConvEmbedding(smi_vocab_size, embedding_size, conv_filters, output_dim, 'seq')
        self.resnet = ResNet(embedding_size, n_layer)  
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layer)])

    def forward(self, smi_input):
        output_emb = self.emb(smi_input)
        output_emb = self.resnet(output_emb)  # Pass through ResNet blocks
        enc_self_attn_mask = get_attn_pad_mask(smi_input, smi_input)
        for layer in self.layers:
            output_emb = layer(output_emb, enc_self_attn_mask)
        return output_emb

class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq_encoder = Seq_Encoder()
        self.smi_encoder = Smi_Encoder()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Squeeze(),
            nn.Linear(1024, 256),
            nn.Dropout(0.5),
            nn.PReLU(),
            nn.Linear(256, 64),
            nn.Dropout(0.5),
            nn.PReLU(),
            nn.Linear(64, 1),
            Squeeze())

    def forward(self, seq_encode, smi_encode, poc_encode):
        seq_outputs = self.seq_encoder(seq_encode, poc_encode)
        smi_outputs = self.smi_encoder(smi_encode)
        score = torch.matmul(seq_outputs, smi_outputs.transpose(-1, -2)) / np.sqrt(embedding_size)
        final_outputs = self.fc(score)
        return final_outputs
