import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TokenAndPositionEmbedding(nn.Module):

    def __init__(self, max_length, vocab_size, embed_dim):
        super().__init__()
        self.token_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        self.position_embedding = nn.Embedding(num_embeddings=max_length, embedding_dim=embed_dim)

    def forward(self, x):
        positions = torch.arange(start=0, end=x.shape[1], step=1).to(DEVICE)
        position_embedding = self.position_embedding(positions)
        token_embedding = self.token_embedding(x)
        return token_embedding + position_embedding
    
class TransformerBlock(nn.Module):

    def __init__(self,
                 num_heads,
                 embd_dim,
                 ff_dim,
                 dropout_rate=0.1
                 ):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=embd_dim,
            num_heads=num_heads,
            batch_first=True
            )
        self.dropout1 = nn.Dropout(dropout_rate)
        self.norm1 = nn.LayerNorm(embd_dim, eps=1e-6)
        self.ffn_1 = nn.Linear(embd_dim, ff_dim)
        self.ffn_2 = nn.Linear(ff_dim, embd_dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.norm2 = nn.LayerNorm(embd_dim, eps=1e-6)

    def generate_causal_mask(self, seq_len):
        return torch.triu(torch.ones(seq_len, seq_len, device=DEVICE) * float('-inf'), diagonal=1)

    def forward(self, batch):
        attention_mask = self.generate_causal_mask(batch.shape[1])
        attention_output, _ = self.attention(batch, batch, batch, attn_mask=attention_mask)
        attention_output = self.dropout1(attention_output)
        attention_output = self.norm1(attention_output + batch)
        ffn_output = F.gelu(self.ffn_1(attention_output))
        ffn_output = self.ffn_2(ffn_output)
        ffn_output = self.dropout2(ffn_output)
        ffn_output = self.norm2(ffn_output + attention_output)
        return ffn_output
    
class MidiGPT(nn.Module):
    def __init__(
            self,
            max_length: int,
            vocab_size: int,
            embed_dim: int,
            feed_forward_dim: int,
            num_heads: int,
        ):
        super().__init__()
        self.embedding_layer = TokenAndPositionEmbedding(max_length, vocab_size, embed_dim)
        self.transformer_block = TransformerBlock(num_heads, embed_dim, feed_forward_dim)
        self.output_layer = nn.Linear(embed_dim, vocab_size)

    def forward(self, batch):
        embedding = self.embedding_layer(batch)
        transformer_output = self.transformer_block(embedding)
        output = self.output_layer(transformer_output)
        return output