import torch
from torch import nn
import math
device = "cuda" if torch.cuda.is_available() else "cpu"


class MultiHeadAttention(nn.Module):
  def __init__(self, d_model, n_heads):
    super().__init__()

    assert d_model % n_heads == 0

    self.d_model = d_model
    self.n_heads = n_heads
    self.head_dim = d_model // n_heads

    self.query = nn.Linear(in_features=d_model, out_features=d_model).to(device)
    self.key = nn.Linear(in_features=d_model, out_features=d_model).to(device)
    self.value = nn.Linear(in_features=d_model, out_features=d_model).to(device)

    self.out_proj = nn.Linear(d_model, d_model).to(device)

  def forward(self, X):
    X = X.to(device)
    B, T, C = X.shape
    Q, K, V = self.query(X), self.key(X), self.value(X)

    Q = Q.view(B, T, self.n_heads, self.head_dim).to(device)
    K = K.view(B, T, self.n_heads, self.head_dim).to(device)
    V = V.view(B, T, self.n_heads, self.head_dim).to(device)

    Q = Q.transpose(1, 2)
    K = K.transpose(1, 2)
    V = V.transpose(1, 2)

    scores = Q @ K.transpose(-1, -2)
    scores = scores / math.sqrt(self.head_dim)


    causal_mask = torch.tril(torch.ones(T, T)).to(device)
    scores = scores.masked_fill(causal_mask == 0, float("-inf")).to(device)

    attn_weights = torch.softmax(scores, dim = -1).to(device)

    attn = attn_weights @ V


    attn = attn.transpose(1, 2)

    attn = attn.contiguous().view(B, T, self.d_model).to(device)

    attn = self.out_proj(attn)
    return attn




class MLP(nn.Module):
  def __init__(self, input_dimensions, hidden_layers, output_dimensions) -> None:
    super().__init__()

    self.mpl_layer = nn.Sequential(
        nn.Linear(in_features=input_dimensions, out_features=hidden_layers).to(device),
        nn.GELU().to(device),
        nn.Linear(in_features=hidden_layers, out_features=output_dimensions).to(device)
    ).to(device)


  def forward(self, X):
    X = X.to(device)
    return self.mpl_layer(X)


class TransformerBlock(nn.Module):
  def __init__(self, d_model, n_heads, hidden_layers):
    super().__init__()

    self.multi_head_attention = MultiHeadAttention(d_model, n_heads).to(device)
    self.mlp = MLP(d_model, hidden_layers, d_model).to(device)

    self.ln1 = nn.LayerNorm(d_model).to(device)
    self.ln2 = nn.LayerNorm(d_model).to(device)


  def forward(self, X):
    X = X.to(device)

    normalized_X = self.ln1(X)
    attention_weights = self.multi_head_attention(normalized_X)
    attention_weights = X + attention_weights


    normalized_attention_weights = self.ln2(attention_weights).to(device)
    mlp_output = self.mlp(normalized_attention_weights)

    output = attention_weights + mlp_output

    return output




class CustomGPT(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.token_embeddings = nn.Embedding(config.vocab_size, config.d_model)

    self.positional_embeddings = nn.Embedding(config.block_size, config.d_model)


    self.transformer_blocks = nn.ModuleList(
        [
            TransformerBlock(config.d_model, config.n_heads, config.hidden_layers)
            for _ in range(config.n_layers)

        ]
    )


    self.layer_norm = nn.LayerNorm(config.d_model)
    self.lm_head = nn.Linear(config.d_model, config.vocab_size)
    self.config = config


  def forward(self, idx):
    B, T = idx.shape

    tok_embd = self.token_embeddings(idx)
    pos = torch.arange(T, device=device)

    pos_embd = self.positional_embeddings(pos)

    X = tok_embd + pos_embd

    for block in self.transformer_blocks:
      X = block(X)

    X = self.layer_norm(X)

    logits = self.lm_head(X)

    return logits


  @torch.no_grad()
  def generate(self, idx, max_new_tokens=500):
    for _ in range(max_new_tokens):
      idx_cont = idx[:, -self.config.block_size:]

      logits = self(idx_cont)

      logits = logits[:, -1, :]

      probs = torch.softmax(logits, dim=-1)
      new_tokens = torch.multinomial(probs, num_samples=1)
      idx = torch.cat((idx, new_tokens), dim=1)

    return idx


class GPTConfig:
  def __init__(self, vocab_size, block_size, d_model, hidden_layers, n_heads, n_layers):
    self.vocab_size = vocab_size
    self.block_size = block_size
    self.d_model = d_model
    self.hidden_layers = hidden_layers
    self.n_heads = n_heads
    self.n_layers = n_layers
    
    # Approximately:
    # vocab_size = tokenizer.n_vocab
    # block_size = 256

    # d_model = 256
    # hidden_layers = 1024
    # n_heads = 4
    # n_layers = 6


# ----------- Data loader ------------
from torch.utils.data import Dataset, DataLoader

class SynthDataset(Dataset):
  def __init__(self, data, block_size):
    self.data = data
    self.block_size = block_size

  def __len__(self):
    return len(self.data) - self.block_size

  def __getitem__(self, idx):
    x = self.data[idx:idx+self.block_size]
    y = self.data[idx+1:idx+self.block_size+1]
    return x, y

def create_dataloaders(tokens:list, train_split:float, device:str, block_size:int, batch_size:int):
    data = torch.tensor(tokens, dtype=torch.long, device=device)
    print("Data length:", len(data))
    
    
    train_split = int(train_split*len(data))
    train_data = data[:train_split]
    test_data = data[train_split:]
    print("Train data length:", len(train_data))
    print("Test data length:", len(test_data))

    train_dataset = SynthDataset(train_data, block_size)
    test_dataset = SynthDataset(test_data, block_size)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    print("Train batches:", len(train_data)/len(train_dataloader))
    print("Test batches:", len(test_data)/len(test_dataloader))
    return train_dataloader, test_dataloader