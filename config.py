from dataclasses import dataclass

@dataclass
class DiTConfig:
    image_size: int = 256
    patch_size: int = 16
    in_channels: int = 3
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 12
    dropout: float = 0.1

    @property
    def n_ctx(self):
        return (self.image_size // self.patch_size) ** 2
