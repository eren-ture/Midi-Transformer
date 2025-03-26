import os
import torch
import torch.nn as nn
import torch.functional as F
from torch.optim import AdamW

from model import MidiGPT
from data import get_dataloader

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

## PARAMETERS ##
# Constants
VOCAB_SIZE = 415

# Train Params
TRAIN_DATA_PATH = './data/train.pt'
SAVE_PATH = './checkpoints'
EPOCHS = 5
PRINT_STEPS = 50
LR = 1e-4
BATCH_SIZE = 4
CHUNK_SIZE = 8000

# Hyperparameters


def main():

    epochs = EPOCHS
    save_path = SAVE_PATH
    os.makedirs(save_path, exist_ok=True)

    # Initialize dataloader
    dataloader = get_dataloader(TRAIN_DATA_PATH, BATCH_SIZE, CHUNK_SIZE)
    print('Initialized dataloader...')

    # Initialize model
    print('Initializing model...')
    model = MidiGPT(
        max_length=CHUNK_SIZE,
        vocab_size=VOCAB_SIZE,
        embed_dim=256,
        feed_forward_dim=1024,
        num_heads=4
        ).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss(ignore_index=414)
    print('Initialized model.')

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        total_loss = 0
        
        model.train()
        for batch_idx, (x, y) in enumerate(dataloader):
            x = x.to(DEVICE)           # Shape: [B, L]
            y = y.to(DEVICE)           # Shape: [B, L]

            optimizer.zero_grad()

            logits = model(x)          # Output: [B, L, vocab_size]
            logits = logits.view(-1, logits.size(-1))  # [B*L, vocab]
            y = y.view(-1)                         # [B*L]

            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (batch_idx + 1) % PRINT_STEPS == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f"  Step {batch_idx+1}/{len(dataloader)} - Loss: {avg_loss:.4f}")

        # Save model checkpoint at end of epoch
        torch.save(model.state_dict(), os.path.join(save_path, 'last.pt'))
        print(f"Model saved to {save_path}, Loss: {total_loss / len(dataloader)}")

if __name__ == '__main__':
    main()