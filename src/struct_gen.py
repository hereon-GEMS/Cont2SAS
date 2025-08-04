# One line to give the program's name and an idea of what it does.
# Copyright (C) 2025  Arnab Majumdar

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program; if not, see
# <https://www.gnu.org/licenses/>.

"""
This script creates mesh
and 
saves it in the data folder

Created on Fri Jun 23 10:28:09 2023

@author: Arnab Majumdar
"""
import sys
import os
import time
import xml.etree.ElementTree as ET

# find current dir and and ..
lib_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# add .. in path
sys.path.append(lib_dir)
# lib imports
from lib import struct_gen as sg # pylint: disable=import-error, wrong-import-position

#timer counter initial
tic = time.perf_counter()

##########################
# read input from xml file
##########################

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

#########################
# create folder structure
#########################

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
        print(f'structure exists at {res_dir}')
        print('!abort!')
        sys.exit()
    else:
        print(f'structure exists at {res_dir} but figures will be updated')
        if plot_node:
            print(f'node figure exists at {res_dir} but will be updated')
        if plot_cell:
            print(f'cell figure exists at {res_dir} but will be updated')
        if plot_mesh:
            print(f'mesh figure exists at {res_dir} but will be updated')
else:
    is_dir=False
    print(f'new structure will be created at {res_dir}')
    if plot_node:
        print(f'node figure will be created at {res_dir}')
    if plot_cell:
        print(f'cell figure will be created at {res_dir}')
    if plot_mesh:
        print(f'mesh figure will be created at {res_dir}')

##################################
# if exist: read structure
# if not exist: generate structure
##################################

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


################
# Create figures
################

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
print(f'Total time taken is {tcomp}s')
