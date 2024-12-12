#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This creates a 3D strcuture and saves it in the data folder

Created on Fri Jun 23 10:28:09 2023

@author: amajumda
"""
import sys
import os
lib_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(lib_dir)

from lib import struct_gen as sg
from lib import plotter as pltr
from lib import simulation as sim
from lib import processing as procs
from lib import datasaver as dsv



import os
import time
import argparse
import sys
import xml.etree.ElementTree as ET

#timer counter initial
tic = time.perf_counter()

"""
read input from xml file
"""

xml_folder='../xml/'

struct_xml=os.path.join(xml_folder, 'struct.xml')

tree=ET.parse(struct_xml)
root = tree.getroot()

# box side lengths
length_a=float(root.find('lengths').find('x').text) 
length_b=float(root.find('lengths').find('y').text)
length_c=float(root.find('lengths').find('z').text)
# number of cells in each direction
nx=int(root.find('num_cell').find('x').text)
ny=int(root.find('num_cell').find('y').text)
nz=int(root.find('num_cell').find('z').text)

"""
create folder structure
"""

# decision paramters
# True: current saved data are updated (if available)
# False: current saved data are not updated (if available)
update=False

# True: new figures created
# False: new figures not created
# three figs: points at nodes,points at cells,Line figure showing mesh connectivity 
plot_node = False
plot_cell = False
plot_mesh = False

# create folders 
os.makedirs('../data', exist_ok=True)

# save length values as strings
# decimal points are replaced with p
length_a_str=str(length_a).replace('.','p')
length_b_str=str(length_a).replace('.','p')
length_c_str=str(length_a).replace('.','p')

# save num_cell values as strings
nx_str=str(nx)
ny_str=str(ny)
nz_str=str(nz)
res_dir=os.path.join('../data/',
                        length_a_str+'_'+length_b_str+'_'+length_c_str+'_'+
                        nx_str+'_'+ny_str+'_'+nz_str)


if os.path.exists(res_dir):
    is_dir=True
    if not update:
        print('structure exists at {0}'.format(res_dir))
        print('!abort!')
        sys.exit()



if is_dir:
    filename='mesh.h5'
    os.makedirs(res_dir, exist_ok=True) 
    nodes, cells, con = dsv.mesh_read(os.path.join(res_dir, filename))
else:
    # generattion of node cell and connectivity
    
    nodes, cells, con = sg.node_cell_gen_3d(length_a, length_b, length_c, nx, ny, nz)
    
    # creation of mesh file data.h5

    filename='data.h5'
    os.makedirs(res_dir, exist_ok=True)
    dsv.mesh_write(os.path.join(res_dir, filename), nodes, cells, con)

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
