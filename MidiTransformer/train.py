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
VAL_DATA_PATH = './data/validation.pt'
PREV_CHECKPOINT = './checkpoints/best.pt'
SAVE_PATH = './checkpoints'
EPOCHS = 100
PRINT_STEPS = 50
LR = 1e-4
BATCH_SIZE = 4
CHUNK_SIZE = 8000

# Hyperparameters


def main():

    epochs = EPOCHS
    save_path = SAVE_PATH
    os.makedirs(save_path, exist_ok=True)

    train_dataloader = get_dataloader(TRAIN_DATA_PATH, BATCH_SIZE, CHUNK_SIZE)
    val_dataloader = get_dataloader(VAL_DATA_PATH, BATCH_SIZE, CHUNK_SIZE, shuffle=False)
    print('Initialized dataloader.')

    print('Initializing model...')
    model = MidiGPT(
        max_length=CHUNK_SIZE,
        vocab_size=VOCAB_SIZE,
        embed_dim=256,
        feed_forward_dim=1024,
        num_heads=4
        ).to(DEVICE)
    
    if PREV_CHECKPOINT:
        model.load_state_dict(torch.load(PREV_CHECKPOINT, map_location=DEVICE))
        print(f"Loaded checkpoint from {PREV_CHECKPOINT}")
    
    optimizer = AdamW(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss(ignore_index=414)
    best_val_loss = float('inf')

    print('Initialized model.')

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        train_total_loss = 0
        
        model.train()
        for batch_idx, (x, y) in enumerate(train_dataloader):
            x = x.to(DEVICE)           # Shape: [B, L]
            y = y.to(DEVICE)           # Shape: [B, L]

            optimizer.zero_grad()

            logits = model(x)          # Output: [B, L, vocab_size]
            logits = logits.view(-1, logits.size(-1))  # [B*L, vocab]
            y = y.view(-1)                         # [B*L]

            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            train_total_loss += loss.item()

            if (batch_idx + 1) % PRINT_STEPS == 0:
                avg_loss = train_total_loss / (batch_idx + 1)
                print(f"  Step {batch_idx+1}/{len(train_dataloader)} - Loss: {avg_loss:.4f}")

        val_total_loss = 0
        model.eval()
        for batch_idx, (x, y) in enumerate(val_dataloader):
            x = x.to(DEVICE)           # Shape: [B, L]
            y = y.to(DEVICE)           # Shape: [B, L]

            with torch.no_grad():
                logits = model(x)          # Output: [B, L, vocab_size]
                logits = logits.view(-1, logits.size(-1))
                y = y.view(-1)                         # [B*L]

            loss = loss_fn(logits, y)
            val_total_loss += loss.item()

        print(f"Train Loss: {train_total_loss / len(train_dataloader):.4f}, Validation Loss: {val_total_loss / len(val_dataloader):.4f}")

        # Save model checkpoint at end of epoch
        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            torch.save(model.state_dict(), os.path.join(save_path, 'best.pt'))
            print(f"Best model saved to {save_path}")
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(save_path, f'epoch_{epoch+1}.pt'))
            print(f"Model saved to {save_path}")

if __name__ == '__main__':
    main()