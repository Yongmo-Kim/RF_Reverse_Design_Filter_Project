# -*- coding: utf-8 -*-
"""
Custome Layout Generator v3 (Custom Pattern)
- 모자이크 크기: 25x25 (10x10mm)
- 픽셀 크기: 0.4mm
- 기판 크기: 20x20mm
- 사용자가 지정한 25x25 배열 기반 생성
- 대각선 제거 로직 적용
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

CUSTOM_PATTERN = [
    "0111010011110101011010011",
    "1011011100101100111110100",
    "0010010011101011000111001",
    "1000000110111111111111010",
    "0001101010001011110001110",
    "0101100000110101101011110",
    "0110000100101011010110010",
    "0101001101011111010010100",
    "0010101001111110000011101",
    "0010101001100011010000110",
    "1100100010011111110000010",
    "1000100011000010010101011",
    "0010111111000000100111101",
    "0011111000010101010001001",
    "1011110101011001100011010",
    "0001011111001010110011110",
    "0010010010111010000010101",
    "0110010100111111010111100",
    "1001100111111110001110110",
    "0000001010101101001000110",
    "0000101011000100110011101",
    "0110100110001110001101100",
    "1001111100010111101100111",
    "0101000001001001101010001",
    "1110000100100100000100001"
]



@dataclass
class CustomeConfig:
    board_size: float = 20.0
    pixel_boundary: float = 10.0
    boundary_offset: float = 5.0
    pixel_size: float = 0.4
    
    port_width: float = 1.2
    port_length: float = 3.0
    port_y: float = 10.0  # 기판 정중앙
    
    @property
    def grid_size(self) -> int:
        return 25
    
    @property
    def bd_left(self) -> float:
        return self.boundary_offset
        
    @property
    def bd_right(self) -> float:
        return self.boundary_offset + self.pixel_boundary

@dataclass
class Pixel:
    x: float
    y: float
    size: float
    pixel_id: int = 0

@dataclass
class LayoutData:
    pixels: List[Pixel] = field(default_factory=list)
    triangles: list = field(default_factory=list)
    config: CustomeConfig = field(default_factory=CustomeConfig)
    grid: Optional[np.ndarray] = None

class CustomePixelGenerator:
    def __init__(self, config: CustomeConfig):
        self.cfg = config
        self.grid = np.zeros((self.cfg.grid_size, self.cfg.grid_size), dtype=bool)

    def generate(self) -> LayoutData:
        layout = LayoutData(config=self.cfg)
        
        # 커스텀 패턴 리딩 (Y축은 텍스트 형태에서 위가 [0]이므로 물리 좌표계에 맞게 반전하여 치환)
        for row_idx, row_str in enumerate(CUSTOM_PATTERN):
            for col_idx, char in enumerate(row_str):
                grid_x = col_idx
                grid_y = (self.cfg.grid_size - 1) - row_idx
                
                if char == '1':
                    self.grid[grid_x, grid_y] = True
                    
        layout.triangles = [] # 룰 적용사항, 대각선 제거
        layout.pixels = self._grid_to_pixels()
        layout.grid = self.grid.copy()
        
        return layout

    def _grid_to_pixels(self) -> List[Pixel]:
        xs, ys = np.where(self.grid)
        pixels = []
        for i, (gx, gy) in enumerate(zip(xs, ys)):
            px = self.cfg.boundary_offset + gx * self.cfg.pixel_size
            py = self.cfg.boundary_offset + gy * self.cfg.pixel_size
            pixels.append(Pixel(px, py, self.cfg.pixel_size, i))
        return pixels

if __name__ == "__main__":
    print("Custome Layout Generator V3 (Custom Pattern) Test")
    config = CustomeConfig()
    generator = CustomePixelGenerator(config)
    layout = generator.generate()
    print(f"Pixels generated: {len(layout.pixels)}")
    print(f"Chamfer triangles: {len(layout.triangles)}")
