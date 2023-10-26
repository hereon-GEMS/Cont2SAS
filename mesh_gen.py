#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This creates a 3D strcuture and saves it in the data folder

Created on Fri Jun 23 10:28:09 2023

@author: amajumda
"""

from lib import struct_gen as sg
from lib import plotter as pltr
from lib import simulation as sim
from lib import processing as procs
from lib import datasaver as dsv



import os
import time
import argparse
import sys

#timer counter initial
tic = time.perf_counter()

"""
configuration of the structure 
"""

#box side lengths
length_a=20
length_b=20
length_c=20
# number of cells in each direction
nx=40
ny=40
nz=40


# control parameters 
update=True

# plot values
plot_node = False
plot_cell = False
plot_mesh = True

#folder structure
os.makedirs('data', exist_ok=True)
res_dir=os.path.join('./data/',
                        str(length_a)+'_'+str(length_b)+'_'+str(length_c)+'_'+
                        str(nx)+'_'+str(ny)+'_'+str(nz))

if os.path.exists(res_dir):
    is_dir=True
    if not update:
        print('structure exists at {0}'.format(res_dir))
        print('!abort!')
        sys.exit()



if is_dir:
    filename='data.h5'
    os.makedirs(res_dir, exist_ok=True) 
    nodes, cells, con = dsv.mesh_read(os.path.join(res_dir, filename))
else:
    # generattion of node cell and connectivity
    
    nodes, cells, con = sg.node_cell_gen_3d(length_a, length_b, length_c, nx, ny, nz)
    
    # creation of mesh file data.h5

    filename='data.h5'
    os.makedirs(res_dir, exist_ok=True)
    dsv.mesh_gen(os.path.join(res_dir, filename), nodes, cells, con)

# images of nodes , cells , mesh

os.makedirs(os.path.join(res_dir, 'images'), exist_ok=True)
if plot_node:
    pltr.plotter_3d(nodes,save_plot=True,save_dir=os.path.join(res_dir, 'images'),
                filename='node', figsize=(10,10))
if plot_cell:
    pltr.plotter_3d(cells,save_plot=True,save_dir=os.path.join(res_dir, 'images'),
                filename='cell', figsize=(10,10))
if plot_mesh:
    pltr.mesh_plotter_3d(nodes, con, save_plot=True,save_dir=os.path.join(res_dir, 'images'),
                filename='mesh', figsize=(10,10))
