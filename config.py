from dataclasses import dataclass

@dataclass
class Config:
    batch_size: int = 100
    learning_rate: float = 0.001
    epochs: int = 100
    n_embeddings: int = 100
    n_heads: int = 10
    n_layers: int = 10
    n_vocab: int = 10000
    n_ctx: int = 1024
    n_embd: int = 1024
    n_head: int = 10
    n_layer: int = 10
    n_vocab: int = 10000