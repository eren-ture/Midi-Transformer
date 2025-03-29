import numpy as np
import torch
from model import MidiGPT
from midi_tools import tensor_to_midi

OUTPUT_MIDI_PATH = './outputs/output.mid'
MAX_TOKENS = 1000
TEMPERATURE = 0.1

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():

    def sample_from(probs: np.ndarray, temp: float) -> int:
        """Sample a token from the list of probabilities."""
        probs = probs ** (1 / temp)
        probs = probs / np.sum(probs)
        return np.random.choice(len(probs), p=probs)
    
    model = MidiGPT(
        max_length=8000,
        vocab_size=415,
        embed_dim=256,
        feed_forward_dim=1024,
        num_heads=4
    ).to(DEVICE)
    
    with torch.no_grad():
        start_tokens = [413]
        for _ in range(MAX_TOKENS):
            x = torch.tensor([start_tokens]).to(DEVICE)
            y = model(x)
            next_token = sample_from(
                torch.nn.functional.softmax(y[0][-1], dim=0).cpu().numpy(),
                TEMPERATURE,
            )
            if next_token == 414:
                break
            start_tokens.append(next_token)
    
    tensor_to_midi(
        start_tokens,
        ppq=480,
        tempo=120,
        output_midi_path=OUTPUT_MIDI_PATH,
    )
    print(f"MIDI file generated: {OUTPUT_MIDI_PATH}")


if __name__ == "__main__":
    main()