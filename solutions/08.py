import numpy as np
import torch
from torch import nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class CharLevelTokenizerLt:
  def __init__(self):
    self.i2t = [
      ' ',
      'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
      'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
      'r', 's', 't', 'u', 'v', 'y', 'z', 'č',
      'ė', 'š', 'ū', 'ž'
    ]
    self.t2i = {token:index for index, token in enumerate(self.i2t)}
    self.no_tokens = len(self.i2t)
    self.space_idx = 0

  def string_to_idx(self, string):
    tokens = [token for token in string]
    return self.tokens_to_idx(tokens)

  def tokens_to_idx(self, tokens):
    return [self.t2i[token] for token in tokens]

  def idx_to_string(self, indices):
    tokens = self.idx_to_tokens(indices)
    return ' '.join(tokens)

  def idx_to_tokens(self, indices):
    return [self.i2t[idx] for idx in indices]

class NameData(Dataset):
  def __init__(self):
    self.tokenizer = CharLevelTokenizerLt()
    self.data = pd.read_csv("../data/names/vardai.csv")

    self.no_tokens = self.tokenizer.no_tokens
    self.space_idx = self.tokenizer.space_idx

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    name = self.data.loc[idx]["name"]

    inp  = [self.space_idx] + self.tokenizer.tokens_to_idx(name) 
    inp = torch.IntTensor(inp)

    target = self.tokenizer.tokens_to_idx(name) + [self.space_idx]
    target = torch.LongTensor(target)
    target = (
      torch.zeros(self.no_tokens*len(target), dtype=torch.float)
        .view(-1, self.no_tokens)
        .scatter(1, target.view(-1, 1), value=1)
    )

    return inp, target

data = NameData()

def collate_names(inp):
  X = torch.cat([x for x, _ in inp], dim=0)
  target = torch.cat([y for _, y in inp], dim=0)

  return X.unsqueeze(0), target.unsqueeze(0)

class Generator(nn.Module):
  def __init__(self, embed_dim, no_tokens, no_layers, hidden_size):
    super().__init__()
    self.embedding = nn.Embedding(embedding_dim=embed_dim, num_embeddings=no_tokens)
    self.lstm = nn.LSTM(
      input_size=embed_dim,
      hidden_size=hidden_size,
      num_layers=no_layers,
      batch_first=True,
    )
    self.toprobs = nn.Linear(hidden_size, no_tokens)

  def forward(self, x, inner_state=None):
    x = self.embedding(x)
    if inner_state is None:
      x, inner = self.lstm(x)
    else:
      x, inner = self.lstm(x, inner_state)
    return self.toprobs(x), inner

# Hyperparameters
learning_rate = 0.0001
epochs = 10
batch_size = 32

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

model = Generator(
  no_tokens=data.no_tokens,
  embed_dim=128,
  no_layers=2,
  hidden_size=2048
).to(device)

dataloader = DataLoader(
  data,
  batch_size=batch_size,
  shuffle=True,
  collate_fn=collate_names
)

loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

def train_epoch(dataloader, model, loss_fn, optimizer):
  size = len(dataloader.dataset)

  model.train() # Set model to training mode
  inner_state = None

  for batch, (X, y) in enumerate(dataloader):
    X, y = X.to(device), y.to(device)
    # Forward pass
    pred, _ = model(X)
    loss = loss_fn(pred, y)

    # Backward pass
    loss.backward()
    optimizer.step()

    # Reset the computed gradients back to zero
    optimizer.zero_grad()

    if batch % 50 == 0:
      loss, current = loss.item(), batch * batch_size + len(X)
      print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# Organize the training loop
for t in range(epochs):
  print(f"Epoch {t+1}\n-------------------------------")
  train_epoch(dataloader, model, loss_fn, optimizer)

print("Done!")

def sample(model, seed, temperature=1.0):
  assert len(seed) > 0
  model.eval()
  with torch.no_grad():
    tokenizer = CharLevelTokenizerLt()
    result = ' ' + seed
    inner_state = None
    while True:
      inp = torch.LongTensor(tokenizer.string_to_idx(result)).unsqueeze(0).to(device)
      if inner_state is None:
        inp, inner_state = model(inp)
      else:
        inp, inner_state = model(inp, inner_state)
      inp = inp[0, -1, :]
      p = nn.functional.softmax(inp, dim=0)
      cd = torch.distributions.Categorical(p)
      next_char = cd.sample()
      if next_char == tokenizer.space_idx:
        break
      result += tokenizer.idx_to_string([next_char])

    return result[1:]

print(
  ' '.join([sample(model=model, seed='jo') for _ in range(5)]) + '\n'
  + ' '.join([sample(model=model, seed='jū') for _ in range(5)]) + '\n'
  + ' '.join([sample(model=model, seed='ja') for _ in range(5)]) + '\n'
  + ' '.join([sample(model=model, seed='je') for _ in range(5)]) + '\n'
)