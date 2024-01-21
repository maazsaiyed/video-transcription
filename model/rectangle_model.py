from dataclasses import dataclass, field
from typing import List, Sequence
from model.point_model import Point


@dataclass
class Rectangle:
    """Model storing rectangle coordinates"""

    _top_left: Point
    _bottom_right: Point = field(default=None, init=False)
    _width: int
    _height: int

    def __post_init__(self):
        if not self._bottom_right:
            self._bottom_right = Point(
                x=self._top_left.x + self._width,
                y=self._top_left.y + self._height
            )

    @classmethod
    def from_haarcascade(cls, location: Sequence[int]):
        """Create Rectangle object from the output of haarcascade detections"""
        return Rectangle(
            _top_left=Point(x=location[0], y=location[1]),
            _width=location[2],
            _height=location[3]
        )

    @property
    def get_top_left(self) -> Point:
        return self._top_left

    @property
    def get_bottom_right(self) -> Point:
        return self._bottom_right

    @property
    def get_width(self) -> int:
        return self._width

    @property
    def get_height(self) -> int:
        return self._height
