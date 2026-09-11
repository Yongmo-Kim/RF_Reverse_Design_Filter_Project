#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Random BPF EMerge HPC Simulation (generator_new integration)
기판 20x20mm (1.2mm 두께), 에어박스 40x40mm
랜덤 시드 기반 모자이크 패턴 (대각선 제거)
출력 데이터: S2P, PNG, NPZ
"""

import gmsh
import numpy as np
import argparse
from pathlib import Path
import sys
import shutil
from unittest.mock import MagicMock

# [CRITICAL FIX] HPC 로깅, MPI 에러 방지
sys.modules["mpi4py"] = MagicMock()
sys.modules["mpi4py.MPI"] = MagicMock()
sys.modules["mumps_emerge"] = MagicMock()
sys.modules["mumps_emerge.interface"] = MagicMock()

import emerge as em


from generator_new import BPFConfig, PixelGenerator

def main(seed: int, output_dir: Path = None, visualize: bool = False):
    if output_dir is None:
        output_dir = Path("/scratch/home/jungsu0910/Coral1")
    
    result_dir = output_dir / "result"
    s2p_dir = result_dir / "s2p"
    png_dir = result_dir / "png"
    npz_dir = result_dir / "npz"
    for d in [s2p_dir, png_dir, npz_dir]:
        d.mkdir(exist_ok=True, parents=True)
    
    config = BPFConfig()
    generator = PixelGenerator(config, seed=seed)
    layout = generator.generate()
    
    print(f"Random Layout Generated (20x20 Board)")
    print(f"  Pixels: {len(layout.pixels)}")
    
    mm = 0.001
    th = 1.2            
    er = 4.0            
    tand = 0.013        
    cond_t = 0.017      
    cond_s = 4.1e7      
    
    MAX_FREQ = 30e9
    c = 3e8
    lambda_air = (c / MAX_FREQ) * 1000
    lambda_val = (c / MAX_FREQ) * 1000 / np.sqrt(er) 
    
    MAX_EDGE = lambda_air / 4
    MIN_EDGE = lambda_val / 15
    PCB_EDGE_SIZE = lambda_val / 5
    Hair = 9.375
    
    pcbmat = em.Material(er=er, tand=tand)
    condmat = em.Material(cond=cond_s, name="Gold")
    airmat = em.Material(name="Air")
    
    sim_name = f'bpf_20mm_seed{seed}'
    m = em.Simulation(sim_name, loglevel='WARNING')
    m.check_version("2.1.0")
    m.set_solver(em.EMSolver.PARDISO)
    
    layouter = em.geo.PCB(
        th, unit=mm, material=pcbmat, trace_material=condmat,
        trace_thickness=cond_t * mm, thick_traces=True, layers=2
    )
    
    for px in layout.pixels:
        layouter.add_poly(
            [px.x, px.x + px.size, px.x + px.size, px.x],
            [px.y, px.y, px.y + px.size, px.y + px.size]
        )
    
    PORT_W = config.port_width
    PORT_L = config.port_length
    port_y_min = config.port_y - PORT_W / 2
    port_y_max = config.port_y + PORT_W / 2
    
    port1_x_start = config.bd_left - PORT_L
    layouter.add_poly([port1_x_start, config.bd_left, config.bd_left, port1_x_start],
                      [port_y_min, port_y_min, port_y_max, port_y_max])
    
    port2_x_end = config.bd_right + PORT_L
    layouter.add_poly([config.bd_right, port2_x_end, port2_x_end, config.bd_right],
                      [port_y_min, port_y_min, port_y_max, port_y_max])
    
    polies = layouter.compile_paths(merge=True)
    
    port1_x = config.bd_left - PORT_L
    layouter.new(port1_x, config.port_y, PORT_W, (-1, 0)).store('p1')
    port2_x = config.bd_right + PORT_L
    layouter.new(port2_x, config.port_y, PORT_W, (1, 0)).store('p2')
    
    # PCB 경계 수동 계산 (빈 픽셀로 인한 layouter 자동 축소 방지)
    all_xs = [port1_x_start, config.bd_left, port2_x_end, config.bd_right]
    all_ys = [port_y_min, port_y_max]
    for px in layout.pixels:
        all_xs.extend([px.x, px.x + px.size])
        all_ys.extend([px.y, px.y + px.size])
        
    actual_min_x = min(all_xs)
    actual_max_x = max(all_xs)
    actual_min_y = min(all_ys)
    actual_max_y = max(all_ys)
    
    layouter.determine_bounds(
        leftmargin=actual_min_x - 0.0,
        rightmargin=config.board_size - actual_max_x,
        bottommargin=actual_min_y - 0.0,
        topmargin=config.board_size - actual_max_y
    )
    
    pcb = layouter.generate_pcb(True, merge=True)
    
    ground = em.geo.Box(
        config.board_size * mm, config.board_size * mm, cond_t * mm,
        position=(0, 0, -(th + cond_t) * mm)
    )
    ground.set_material(condmat)
    
    airbox_offset_x = (config.board_size / 2 - 20.0) * mm
    airbox_offset_y = (config.board_size / 2 - 20.0) * mm
    AIRBOX_HEIGHT = (th + cond_t) + Hair
    
    airbox = em.geo.Box(
        40.0 * mm, 40.0 * mm, AIRBOX_HEIGHT * mm,
        position=(airbox_offset_x, airbox_offset_y, -(th + cond_t) * mm)
    )
    airbox.material = airmat
    
    p1 = layouter.lumped_port(layouter.load('p1'))
    p2 = layouter.lumped_port(layouter.load('p2'))
    m.commit_geometry()
    
    freq_list = []
    freq_list.append(0.1e9)
    freq_list.extend(list(np.arange(0.5e9, 1.5e9 + 1e6, 0.5e9)))
    freq_list.extend(list(np.arange(2.0e9, 12.0e9 + 1e6, 0.2e9)))
    freq_list.extend(list(np.arange(12.5e9, 30.0e9 + 1e6, 0.5e9)))
    m.mw.set_frequency(freq_list)
    print(f"📊 Frequency points: {len(freq_list)} (expected: 91)")
    
    m.mesher.set_boundary_size(polies.face('+z'), MIN_EDGE * mm, growth_rate=5)
    m.mesher.set_boundary_size(pcb, PCB_EDGE_SIZE * mm)
    gmsh.option.setNumber("Mesh.MeshSizeMin", MIN_EDGE * mm)
    gmsh.option.setNumber("Mesh.MeshSizeMax", MAX_EDGE * mm)
    m.generate_mesh()
    print("✅ Mesh generated")
    
    if visualize:
        gmsh.fltk.run()
    
    port1 = m.mw.bc.LumpedPort(p1, 1, Z0=50)
    port2 = m.mw.bc.LumpedPort(p2, 2, Z0=50)
    for boundary in [airbox.top, airbox.left, airbox.right, airbox.front, airbox.back]:
        m.mw.bc.AbsorbingBoundary(boundary, order=2, abctype='B')
    
    print(f"Starting HPC simulation for seed={seed} with 16 cores...")
    data = m.mw.run_sweep(parallel=True, n_workers=16, frequency_groups=16, multi_processing=True)
    
    grid = data.scalar.grid
    
    # Export S2P
    s2p_file = s2p_dir / f"seed{seed}.s2p"
    grid.export_touchstone(str(s2p_file), Z0ref=50, format="RI", funit="GHZ")
    print(f"✅ S2P saved: {s2p_file}")
    
    # Export PNG
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        
        fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
        ax.set_facecolor('black')
        for px in layout.pixels:
            rect = Rectangle((px.x, px.y), px.size, px.size, facecolor='gold', edgecolor='none')
            ax.add_patch(rect)
        
        ax.set_xlim(0, config.board_size)
        ax.set_ylim(0, config.board_size)
        ax.set_aspect('equal')
        ax.axis('off')
        
        png_file = png_dir / f"seed{seed}.png"
        fig.savefig(png_file, dpi=100, bbox_inches='tight', pad_inches=0, facecolor='black')
        plt.close(fig)
        print(f"✅ PNG saved: {png_file}")
    except ImportError:
        pass
    
    # Export NPZ
    npz_file = npz_dir / f"seed{seed}.npz"
    np.savez_compressed(
        npz_file,
        grid=np.rot90(layout.grid).astype(np.uint8),
        seed=seed,
        board_size=config.board_size,
        pixel_size=config.pixel_size
    )
    print(f"✅ NPZ saved: {npz_file}")
    
    m.save()
    print(f"✅ Simulation saved: {sim_name}")
    
    result_folder = Path.cwd() / f"{sim_name}.EMResults"
    if result_folder.exists():
        shutil.rmtree(result_folder)
    
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Random Seed BPF Simulation")
    parser.add_argument('--seed', type=int, default=12345, help='Random generator seed')
    parser.add_argument('--output', type=str, default='/scratch/home/jungsu0910/Coral1')
    parser.add_argument('--visualize', action='store_true', help='Open GMSH GUI before simulation')
    args = parser.parse_args()
    main(args.seed, Path(args.output), args.visualize)
