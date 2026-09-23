
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, IterableDataset
import numpy as np

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


class StreamingDataset(IterableDataset):
    def __init__(
        self,
        dataset,
        tokenizer,
        block_size,
        skip_tokens=0,
        tokenize_batch_size=64
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.skip_tokens = skip_tokens
        self.tokenize_batch_size = tokenize_batch_size

    def __iter__(self):
        buffer = []
        seen_tokens = 0

        batch = []

        for sample in self.dataset:

            batch.append(sample["text"])

            # Tokenize multiple documents together
            if len(batch) < self.tokenize_batch_size:
                continue

            token_lists = self.tokenizer.encode_ordinary_batch(batch)
            batch = []

            for tokens in token_lists:

                # Skip tokens from previous phases
                tokens.append(self.tokenizer.eot_token)

              
                if seen_tokens + len(tokens) <= self.skip_tokens:
                    seen_tokens += len(tokens)
                    continue

                # Skip part of a document if skip_tokens
                # falls inside this document
                if seen_tokens < self.skip_tokens:
                    start = self.skip_tokens - seen_tokens
                    tokens = tokens[start:]
                    seen_tokens = self.skip_tokens

                buffer.extend(tokens)

                # Create training samples
                while len(buffer) >= self.block_size + 1:

                    x = buffer[:self.block_size]
                    y = buffer[1:self.block_size + 1]


                    buffer = buffer[self.block_size:]


                    yield (
                        torch.tensor(x, dtype=torch.long),
                        torch.tensor(y, dtype=torch.long)
                    )



class TokenShardDataset(Dataset):
    def __init__(self, path, block_size=256):
        self.tokens = np.load(path, mmap_mode="r")

        self.block_size = block_size


    def __len__(self):
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx):
        x = self.tokens[idx:idx+self.block_size]
        y = self.tokens[idx+1:idx+self.block_size+1]

        # return (
        #     torch.from_numpy(x.astype(np.int64)),
        #     torch.from_numpy(y.astype(np.int64)),
        # )
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


def create_shards_dataloaders(path, train_split, block_size, batch_size, num_workers=0):
    dataset = TokenShardDataset(path, block_size)

    split = int(train_split*len(dataset))

    train_dataset = torch.utils.data.Subset(dataset, range(0, split))
    test_dataset = torch.utils.data.Subset(dataset, range(split, len(dataset)))

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers
    )

    print("Total samples:", len(dataset))
    print("Train samples:", len(train_dataset))
    print("Test samples:", len(test_dataset))

    return train_dataloader, test_dataloader