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

import logging

# Create a logger
logger = logging.getLogger('my_logger')

# Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR)
logger.setLevel(logging.INFO)

# Create a file handler to write log messages to a file
file_handler = logging.FileHandler('ball_in_box.log')

# Create a console handler to display log messages in the console
console_handler = logging.StreamHandler()

# Create a formatter to specify the log message format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Set the formatter for the handlers
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

import os
import time
import argparse


logging.info('Starting generation of ball in box')
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
nx=20
ny=20
nz=20
# ball radius
radius=5
# sld of ball and box (sld_in = ball, sld_out = box) 
sld_in=1
sld_out=0

logging.info('box: ({0}, {1},{2}), number of cells: ({3},{4},{5}), \
             radius of ball: {6}, sld: ball-{7}, box-{8}'\
                 .format(length_a,length_b,length_c,nx,ny,nz,\
                         radius, sld_in, sld_out))

# genration of nodes and connectivity matrix
tc_sg_start = time.perf_counter()
coords_3d,con_3d=sg.struct_gen_3d(length_a,length_b,length_c,nx,ny,nz)
tc_sg_stop = time.perf_counter()
#plotter_3d(coords_3d, show_plot=False)
logging.info('generation of nodes and connectivity matrix took {0} secs'.format(-tc_sg_start+tc_sg_stop))
print('generation of nodes and connectivity matrix took {0} secs'.format(-tc_sg_start+tc_sg_stop))

# generation of sld
tc_sld_start = time.perf_counter()
sld_3d=sim.sph_grain_3d(coords_3d,[length_a/2,length_b/2,length_c/2],radius,sld_in,sld_out)
tc_sld_stop = time.perf_counter()
logging.info('generation of sld took {0} secs'.format(-tc_sld_start+tc_sld_stop))
print('generation of sld took {0} secs'.format(-tc_sld_start+tc_sld_stop))

#colorplot_node_3d(coords_3d,sld_3d,nx,ny,nz,show_plot=False)
#mesh_plotter_3d(coords_3d, con_3d)

# conversion of node o cell
tc_node2cell_start = time.perf_counter()
cell_3d, cell_sld_3d = procs.node2cell_3d (coords_3d,con_3d, [sld_3d], [0], nx, ny, nz)
tc_node2cell_stop = time.perf_counter()
logging.info('conersion of node data to cell data took {0} secs'.format(-tc_node2cell_start+tc_node2cell_stop))
print('conersion of node data to cell data took {0} secs'.format(-tc_node2cell_start+tc_node2cell_stop))

#plot the sld distribution at middle z
pltr.colorplot_cell_3d(cell_3d,cell_sld_3d,nx,ny,nz,show_plot=False)


tc_savedata_start = time.perf_counter()
os.makedirs('data', exist_ok=True)
res_folder=os.path.join('./data/',
                        str(length_a)+'_'+str(length_b)+'_'+str(length_c)+'_'+
                        str(nx)+'_'+str(ny)+'_'+str(ny)+'_'+
                        str(radius)+'_'+str(sld_in)+'in_'+str(sld_out)+'out')
filename='data.h5'
dsv.HDFwriter(coords_3d,con_3d, sld_3d, cell_3d, cell_sld_3d, filename,Folder=res_folder)

tc_savedata_stop = time.perf_counter()
logging.info('saving everything in the h5 file took {0} secs'.format(-tc_savedata_start+tc_savedata_stop))
print('saving everything in the h5 file took {0} secs'.format(-tc_savedata_start+tc_savedata_stop))

toc = time.perf_counter()
logging.info('Whole structure generation took {0} secs'.format(-tic+toc))
print('Whole structure generation took {0} secs'.format(-tic+toc))