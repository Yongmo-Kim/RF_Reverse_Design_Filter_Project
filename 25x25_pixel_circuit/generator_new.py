# -*- coding: utf-8 -*-
"""
Random BPF Generator - 20mm Board Version
Based on user specs for Refactoring

Key Changes:
- board_size: 20.0 mm
- boundary_offset: 5.0 mm
- pixel_size: 0.4 mm
- port width: 1.2 mm, port_y: 10.0 mm
- Removed Separator logic
- Removed Chamfer/Diagonal lines (대각전)
"""

import argparse
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set

import numpy as np

# Optional imports for visualization
try:
    import gmsh
    import emerge as em
except ImportError:
    gmsh = None
    em = None

# ============================================
# Configuration (20mm Board)
# ============================================
@dataclass
class BPFConfig:
    # Board Geometry
    board_size: float = 20.0
    pixel_boundary: float = 10.0
    boundary_offset: float = 5.0
    pixel_size: float = 0.4
    
    # Port configuration
    port_width: float = 1.2
    port_length: float = 3.0
    port_y: float = 10.0
    port_pixel_count: int = 1
    
    # Global density probabilities
    global_ranges: tuple = ((15, 25), (25, 40), (40, 60), (60, 75))
    global_probs: tuple = (0.05, 0.45, 0.4, 0.1)
    
    
    @property
    def grid_size(self) -> int:
        return int(self.pixel_boundary / self.pixel_size)
    
    @property
    def pixel_half(self) -> float:
        return self.pixel_size / 2.0
        

    @property
    def bd_left(self) -> float:
        return self.boundary_offset
        
    @property
    def bd_right(self) -> float:
        return self.boundary_offset + self.pixel_boundary
        
    @property
    def bd_top(self) -> float:
        return self.boundary_offset + self.pixel_boundary
        
    @property
    def bd_bottom(self) -> float:
        return self.boundary_offset
    
    def generate_global_pixel_count(self, total_pixels: int) -> int:
        r = random.random()
        cumulative = 0.0
        for (low, high), prob in zip(self.global_ranges, self.global_probs):
            cumulative += prob
            if r <= cumulative:
                density = random.uniform(low, high) / 100.0
                return int(round(total_pixels * density))
        density = random.uniform(self.global_ranges[-1][0], self.global_ranges[-1][1]) / 100.0
        return int(round(total_pixels * density))

# ============================================
# Data Models
# ============================================
@dataclass
class Pixel:
    x: float
    y: float
    size: float
    pixel_id: int = 0

@dataclass
class LayoutData:
    pixels: List[Pixel] = field(default_factory=list)
    config: BPFConfig = field(default_factory=BPFConfig)
    seed: Optional[int] = None
    grid: Optional[np.ndarray] = None

# ============================================
# Logic: Pixel Generator
# ============================================
class PixelGenerator:
    def __init__(self, config: BPFConfig, seed: int = 42):
        self.cfg = config
        self.seed = seed
        self.grid = np.zeros((self.cfg.grid_size, self.cfg.grid_size), dtype=bool)
        
        random.seed(seed)
        np.random.seed(seed)

    def generate(self) -> LayoutData:
        layout = LayoutData(config=self.cfg, seed=self.seed)
        
        self._fill_global(set())
        
        layout.pixels = self._grid_to_pixels()
        layout.grid = self.grid.copy()
        
        return layout



    def _fill_global(self, port_pixels):
        total_pixels = self.cfg.grid_size * self.cfg.grid_size
        target_pixel_count = self.cfg.generate_global_pixel_count(total_pixels)
        
        available_mask = ~self.grid
        available_indices = np.argwhere(available_mask)
        
        if len(available_indices) == 0: return
        
        available_abs = [(idx[0], idx[1]) for idx in available_indices]
        forced_count = len(port_pixels)
        
        count = target_pixel_count - forced_count
        count = max(0, min(len(available_abs), count))
        
        if count > 0:
            chosen = random.sample(available_abs, count)
            for cx, cy in chosen:
                self.grid[cx, cy] = True

    def _grid_to_pixels(self) -> List[Pixel]:
        xs, ys = np.where(self.grid)
        pixels = []
        for i, (gx, gy) in enumerate(zip(xs, ys)):
            px = self.cfg.boundary_offset + gx * self.cfg.pixel_size
            py = self.cfg.boundary_offset + gy * self.cfg.pixel_size
            pixels.append(Pixel(px, py, self.cfg.pixel_size, i))
        return pixels

# ============================================
# Module Test Context
# ============================================
if __name__ == "__main__":
    print("BPF Generator (20mm Board) Module Test")
    print("=" * 50)
    
    config = BPFConfig()
    print(f"Board Size: {config.board_size}mm")
    print(f"Boundary Offset: {config.boundary_offset}mm")
    print(f"Port Y: {config.port_y}mm with Width: {config.port_width}mm")
    print(f"Grid: {config.grid_size} x {config.grid_size}")
    
    generator = PixelGenerator(config, seed=42)
    layout = generator.generate()
    
    print(f"\nLayout Generated:")
    print(f"  Pixels: {len(layout.pixels)}")
    total_slots = config.grid_size * config.grid_size
    print(f"  Fill Percentage: {len(layout.pixels) / total_slots * 100:.2f}%")
