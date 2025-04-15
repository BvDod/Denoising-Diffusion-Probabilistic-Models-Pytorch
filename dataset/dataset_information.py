from dataclasses import dataclass

@dataclass
class DatasetInfo:
    name: str
    image_size: list[int]
    channels: int
