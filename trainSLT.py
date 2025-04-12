from dataclasses import dataclass

@dataclass
class DiffusionConfig:
    dataset: str
    learning_rate: int


diff_config = DiffusionConfig(
    dataset = "FFHQ",
    learning_rate = 0.001
)

