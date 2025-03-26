import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

def collate_fn(batch, pad_token=414):
    x_batch, y_batch = zip(*batch)
    max_len = max(len(x) for x in x_batch)
    x_padded = [F.pad(x, (0, max_len - len(x)), value=pad_token) for x in x_batch]
    y_padded = [F.pad(y, (0, max_len - len(y)), value=pad_token) for y in y_batch]
    return torch.stack(x_padded), torch.stack(y_padded)

class MidiDataset(Dataset):
    def __init__(self, path, chunk_size):

        self.data = torch.load(path)  # shape: [N, seq_len]
        self.chunk_size = chunk_size
        self.chunks = []

        # TODO: Maybe... Make it so that the chunks don't have "on notes".
        #       One way might be to edit midi_tools.py to return the indices where there are no on_notes.
        for seq_idx, seq in enumerate(self.data):
            seq_len = len(seq)
            num_chunks = (seq_len - 1) // chunk_size
            for i in range(num_chunks):
                start = i * chunk_size
                self.chunks.append((seq_idx, start))

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        seq_idx, start = self.chunks[idx]
        chunk = self.data[seq_idx][start:start + self.chunk_size + 1]  # +1 for y

        x = chunk[:-1]  # input
        y = chunk[1:]   # target
        return x, y

    
def get_dataloader(path, batch_size, chunk_size, shuffle=True):
    return DataLoader(MidiDataset(path, chunk_size), batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)