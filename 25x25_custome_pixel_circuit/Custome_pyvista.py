#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
os.environ.setdefault('PYVISTA_OFF_SCREEN', 'false')

import emerge as em
import numpy as np
import pyvista as pv
import argparse

# V3 커스텀 레이아웃 임포트
from Custome_Layout import CustomeConfig, CustomePixelGenerator

def main(save_path: str = None, offscreen: bool = False):
    if offscreen or save_path:
        pv.OFF_SCREEN = True

    config = CustomeConfig()
    generator = CustomePixelGenerator(config)
    layout = generator.generate()
    
    print(f"Custome Custom Layout V3 Generated")
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
    
    m = em.Simulation('custome_pyvista_v3_custom')
    m.check_version("2.2.0")
    
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
    
    ground = em.geo.Box(config.board_size * mm, config.board_size * mm, cond_t * mm,
                        position=(0, 0, -(th + cond_t) * mm))
    ground.set_material(condmat)
    
    airbox_offset_x = (config.board_size / 2 - 20.0) * mm
    airbox_offset_y = (config.board_size / 2 - 20.0) * mm
    AIRBOX_HEIGHT = (th + cond_t) + Hair
    airbox = em.geo.Box(40.0 * mm, 40.0 * mm, AIRBOX_HEIGHT * mm,
                        position=(airbox_offset_x, airbox_offset_y, -(th + cond_t) * mm))
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
    
    m.mesher.set_boundary_size(polies.face('+z'), MIN_EDGE * mm, growth_rate=5)
    m.mesher.set_boundary_size(pcb, PCB_EDGE_SIZE * mm)
    import gmsh
    gmsh.option.setNumber("Mesh.MeshSizeMin", MIN_EDGE * mm)
    gmsh.option.setNumber("Mesh.MeshSizeMax", MAX_EDGE * mm)
    
    m.generate_mesh()
    print("✅ Mesh generated")
    
    port1 = m.mw.bc.LumpedPort(p1, 1, Z0=50)
    port2 = m.mw.bc.LumpedPort(p2, 2, Z0=50)
    for boundary in [airbox.top, airbox.left, airbox.right, airbox.front, airbox.back]:
        m.mw.bc.AbsorbingBoundary(boundary, order=2, abctype='B')
    
    print("Starting PyVista visualization...")
    m.display._plot.add_mesh(m.display.mesh(pcb), color='#4d7c0f', opacity=0.7, show_edges=True)
    m.display._plot.add_mesh(m.display.mesh(polies), color='#ffc107', opacity=1.0, show_edges=True)
    m.display._plot.add_mesh(m.display.mesh(ground), color='#ffc107', opacity=1.0, show_edges=True)
    m.display._plot.add_mesh(m.display.mesh(airbox), color='#87ceeb', opacity=0.1, show_edges=True)
    
    center = (config.board_size / 2 * mm, config.board_size / 2 * mm, 0)
    m.display._plot.camera_position = [
        (center[0] + 0.05, center[1] + 0.05, 0.1),
        center, (0, 1, 0)
    ]
    m.display._plot.add_axes()
    m.display._plot.set_background('white')

    if save_path:
        m.display._plot.screenshot(save_path)
        print(f"✅ Screenshot saved: {save_path}")
    else:
        print("Opening PyVista window...")
        m.display.show()
    
    print("✅ Done (PyVista Viewer V3 Custom)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Custome PyVista Viewer V3 (Custom)")
    parser.add_argument('--save', type=str, default=None, help='Save screenshot to file')
    parser.add_argument('--offscreen', action='store_true', help='Run in offscreen mode')
    args = parser.parse_args()
    main(save_path=args.save, offscreen=args.offscreen)
