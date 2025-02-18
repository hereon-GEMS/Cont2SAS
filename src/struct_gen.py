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

import time
import xml.etree.ElementTree as ET

#timer counter initial
tic = time.perf_counter()

"""
read input from xml file
"""

xml_folder='./xml/'

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

# element type
el_type=root.find('element').find('type').text
if el_type=='lagrangian':
    el_order=int(root.find('element').find('order').text)
    el_info={'type': el_type,
             'order': el_order}

# decision paramters
# True: current saved figures are updated (if available)
# False: current saved figures are not updated (if available)
update=sg.str_to_bool(root.find('decision').find('update').text)

# True: new figures created
# False: new figures not created
# three figs: points at nodes,points at cells,Line figure showing mesh connectivity 
plot_node = sg.str_to_bool(root.find('decision').find('plot').find('node').text)
plot_cell = sg.str_to_bool(root.find('decision').find('plot').find('cell').text)
plot_mesh = sg.str_to_bool(root.find('decision').find('plot').find('mesh').text)

"""
create folder structure
"""

# create folders 
os.makedirs('./data', exist_ok=True)

# save length values as strings
# decimal points are replaced with p
length_a_str=str(length_a).replace('.','p')
length_b_str=str(length_a).replace('.','p')
length_c_str=str(length_a).replace('.','p')

# save num_cell values as strings
nx_str=str(nx)
ny_str=str(ny)
nz_str=str(nz)

struct_folder_name = (length_a_str + '_' + length_b_str + '_' + length_c_str
                       + '_' + nx_str + '_' + ny_str + '_' + nz_str)
# save elemnt type as string
if el_type=='lagrangian':
    el_order_str=str(el_order)
    struct_folder_name += '_' + el_type + '_' + el_order_str


res_dir=os.path.join('./data/', struct_folder_name +'/structure')


if os.path.exists(res_dir):
    is_dir=True
    if not update:
        print('structure exists at {0}'.format(res_dir))
        print('!abort!')
        sys.exit()
    else:
        print('structure exists at {0} but figures will be updated'.format(res_dir))
        if plot_node:
            print('node figure exists at {0} but will be updated'.format(res_dir))
        if plot_cell:
            print('cell figure exists at {0} but will be updated'.format(res_dir))
        if plot_mesh:
            print('mesh figure exists at {0} but will be updated'.format(res_dir))
else:
    is_dir=False
    print('new structure will be created at {0}'.format(res_dir))
    if plot_node:
        print('node figure will be created at {0}'.format(res_dir))
    if plot_cell:
        print('cell figure will be created at {0}'.format(res_dir))
    if plot_mesh:
        print('mesh figure will be created at {0}'.format(res_dir))


"""
structure read (if exist) or generation (if not exist)
""" 

if is_dir:
    # read structure for updating figures
    struct_file=os.path.join(res_dir, 'struct.h5')
    nodes, cells, con, mesh = sg.mesh_read(struct_file)

else:
    # generattion of node cell and connectivity
    nodes, cells, con, mesh = sg.node_cell_gen_3d(length_a, length_b, length_c, nx, ny, nz, el_info)

    # creation of mesh file struct.h5
    os.makedirs(res_dir, exist_ok=True)
    struct_file=os.path.join(res_dir, 'struct.h5')
    sg.mesh_write(struct_file, nodes, cells, con, mesh)


"""
Create figures
""" 
# images of nodes , cells , mesh
os.makedirs(os.path.join(res_dir, 'figures'), exist_ok=True)
if plot_node:
    sg.plotter_3d(nodes,save_plot=True,save_dir=os.path.join(res_dir, 'figures'),
                filename='node', figsize=(10,10))
if plot_cell:
    sg.plotter_3d(cells,save_plot=True,save_dir=os.path.join(res_dir, 'figures'),
                filename='cell', figsize=(10,10))
if plot_mesh:
    sg.mesh_plotter_3d(nodes, mesh, save_plot=True,save_dir=os.path.join(res_dir, 'figures'),
                filename='mesh', figsize=(10,10))
    
toc = time.perf_counter()
tcomp=toc-tic

print('Program finished running')
print('Total time taken is {0}s'.format(tcomp))