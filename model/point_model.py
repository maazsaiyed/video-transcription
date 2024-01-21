from dataclasses import dataclass
from typing import Tuple


@dataclass
class Point:
    x: int
    y: int

    def get_tuple(self) -> Tuple[int, int]:
        return self.x, self.y
