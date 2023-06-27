#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This creates a 3D strcuture and saves it in the data folder

Created on Fri Jun 23 10:28:09 2023

@author: amajumda
"""

from src import struct_gen as sg
from src import plotter as pltr
from src import simulation as sim
from src import processing as procs
from src import datasaver as dsv

import os
import time

tic = time.perf_counter()

length_a=20
length_b=20
length_c=20
nx=20
ny=20
nz=20
radius=5
sld_in=1
sld_out=0

tic1 = time.perf_counter()
coords_3d,con_3d=sg.struct_gen_3d(length_a,length_b,length_c,nx,ny,nz)
toc1 = time.perf_counter()
print(f"completed coordinate generation in {toc1 - tic1:0.4f} seconds")
#plotter_3d(coords_3d, show_plot=False)
tic1 = time.perf_counter()
sld_3d=sim.sph_grain_3d(coords_3d,[length_a/2,length_b/2,length_c/2],radius,sld_in,sld_out)
toc1 = time.perf_counter()
print(f"completed sld generation in {toc1 - tic1:0.4f} seconds")
#colorplot_node_3d(coords_3d,sld_3d,nx,ny,nz,show_plot=False)
#mesh_plotter_3d(coords_3d, con_3d)
tic1 = time.perf_counter()
cell_3d, cell_sld_3d = procs.node2cell_3d (coords_3d,con_3d, [sld_3d], [0], nx, ny, nz)
toc1 = time.perf_counter()
print(f"converted node data to cell data in {toc1 - tic1:0.4f} seconds")
tic1 = time.perf_counter()
pltr.colorplot_cell_3d(cell_3d,cell_sld_3d,nx,ny,nz,show_plot=False)
toc1 = time.perf_counter()
print(f"plotted sld data in {toc1 - tic1:0.4f} seconds")
tic1 = time.perf_counter()
res_folder=os.path.join('./data/',
                        str(length_a)+'_'+str(length_b)+'_'+str(length_c)+'_'+
                        str(nx)+'_'+str(ny)+'_'+str(ny)+'_'+
                        str(radius)+'_'+str(sld_in)+'in_'+str(sld_out)+'out')
filename='data.h5'
dsv.HDFwriter(coords_3d,con_3d, sld_3d, cell_3d, cell_sld_3d, filename,Folder=res_folder)

toc1 = time.perf_counter()
print(f"saved h5 file in {toc1 - tic1:0.4f} seconds")
toc = time.perf_counter()
print(f"completed everything in {toc - tic:0.4f} seconds")