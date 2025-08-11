# Copyright (C) 2025  Helmholtz-Zentrum-Hereon

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
This generates data for chemical composition change model
data is saved in data folder

Author: Arnab Majumdar
Date: 24.06.2025
"""


import subprocess
import os
import sys
import time
import warnings
import numpy as np

# find current dir and and ..
lib_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
# add .. in path
sys.path.append(lib_dir)
# lib imports
from lib import xml_gen # pylint: disable=wrong-import-position

# ignore warnings

warnings.filterwarnings("ignore")

##############
# input values
##############
# script dir and working dir
script_dir = "src"  # Full path of the script
working_dir = "."  # Directory to run the script in


### struct gen ###
xml_dir='./xml'
length_a=40.
length_b=length_a
length_c=length_a
nx=40
ny=nx
nz=nx
el_type='lagrangian'
el_order=1
update_val=True
plt_node=False
plt_cell=False
plt_mesh=False

### sim gen ###
sim_model='sld_grow'
dt=1
t_end=10
n_ensem=1

### model param ###
rad=10
sld_in_0=2
sld_in_end=5
sld_out=1
qclean_sld=sld_out

### scatt cal ###
num_cat=101 # also check with 3
method_cat='extend'
sassena_exe= '/home/amajumda/Documents/Softwares/sassena/compile/sassena'
mpi_procs=1
num_threads=1
sig_file='signal.h5'
scan_vec=np.array([1, 0, 0])
# these values are taken from publication
# doi: https://doi.org/10.3233/JNR-190116
Q_range=np.array([0.0029, 0.05])
num_points=100
num_orientation=10

### sig_eff cal ###
# distance : distance between detenctor and sample
# wl : neutron wave length
# beam_center_coord : beam center off set from detector center
instrument='SANS-1'
facility='MLZ'
distance=20
wl=4.5
beam_center_coord=np.array([0, 0, 0])

### run vars ###
struct_gen_run=1
sim_run=1
scatt_cal_run=1
sig_eff_cal_run=1

"""
xml gen and run scripts
"""
part_str='Part>>>  '
info_str='Info>>>  '

print('-------------------------------------------------')
print(part_str + 'Generating mesh')
print('')
# time counter start - struct gen
tic_sg = time.perf_counter()
if struct_gen_run==1:
    # generate xml for struct_gen
    print(info_str + 'Generating struct_gen.xml')
    xml_gen.struct_xml_write(xml_dir,
                              length_a, length_b, length_c,
                                nx, ny, nz,
                                  el_type, el_order,
                                    update_val, plt_node, plt_cell, plt_mesh)

    # generate structure
    print(info_str + 'Executing struct_gen.py')
    script_path = os.path.join(script_dir, 'struct_gen.py')  # Full path of the script
    #working_dir = "."  # Directory to run the script in
    subprocess.run(["python", script_path], cwd=working_dir, stderr=subprocess.PIPE, check=True)
else:
    print(info_str + 'structure generation not attempted')
# time counter end - struct gen
toc_sg = time.perf_counter()
ttot_sg= round(toc_sg - tic_sg,3)
print(f'Time taken for generating mesh: {ttot_sg} s')
print('-------------------------------------------------')

print('-------------------------------------------------')
print(part_str + 'Assigning SLD values to nodes')
print('')
# time counter start - simulation gen
tic_sim = time.perf_counter()
if sim_run==1:
    # simulation xml
    print(info_str + 'Generating simulation.xml')
    xml_gen.sim_xml_write(xml_dir, sim_model,dt, t_end, n_ensem)

    # model xml
    print(info_str + 'Generating model_sld_grow.xml')
    #xml_gen.model_ball_xml_write(xml_dir, rad, sld)
    xml_gen.model_sld_grow_xml_write(xml_dir, rad, sld_in_0, sld_in_end, sld_out, qclean_sld)

    # generate structure
    print(info_str + 'Executing sim_gen.py')
    script_path = os.path.join(script_dir, 'sim_gen.py')  # Full path of the script
    #working_dir = "."  # Directory to run the script in
    subprocess.run(["python", script_path], cwd=working_dir, stderr=subprocess.PIPE, check=True)
else:
    print(info_str + 'simulation not attempted')
# time counter end - simulation gen
toc_sim = time.perf_counter()
ttot_sim= round(toc_sim - tic_sim,3)
print(f'Time taken for assigning SLD values: {ttot_sim} s')
print('-------------------------------------------------')

print('-------------------------------------------------')
print(part_str + 'Calculating SAS pattern')
print('')
# time counter start - SAS calculation
tic_scatt = time.perf_counter()
if scatt_cal_run==1:
    # scatt_cal xml
    print(info_str + 'Generating scatt_cal.xml')
    xml_gen.scatt_cal_xml_write(xml_dir, num_cat, method_cat,
                                sassena_exe, mpi_procs, num_threads,
                                sig_file, scan_vec, Q_range,
                                num_points, num_orientation)

    # calculate scattering function
    print(info_str + 'Executing scatt_cal.py')
    script_path = os.path.join(script_dir, 'scatt_cal.py')  # Full path of the script
    #working_dir = "."  # Directory to run the script in
    subprocess.run(["python", script_path], cwd=working_dir, stderr=subprocess.PIPE, check=True)
else:
    print(info_str + 'Calculation of scattring function not attempted')
# time counter end - SAS calculation
toc_scatt = time.perf_counter()
ttot_scatt= round(toc_scatt - tic_scatt,3)
print(f'Time taken for calculating SAS pattern: {ttot_scatt} s')
print('-------------------------------------------------')

print('-------------------------------------------------')
print(part_str + 'Calculating effective cross-section')
print('')
# time counter start - sig_eff calculation
tic_sig_eff = time.perf_counter()
if sig_eff_cal_run==1:
    # scatt_cal xml
    print(info_str + 'Generating sig_eff_cal.xml')
    xml_gen.sig_eff_xml_write(xml_dir, instrument, facility,
                               distance, wl, beam_center_coord)

    # calculate scattering function
    print(info_str + 'Executing sig_eff.py')
    script_path = os.path.join(script_dir, 'sig_eff_cal.py')  # Full path of the script
    #working_dir = "."  # Directory to run the script in
    subprocess.run(["python", script_path], cwd=working_dir, stderr=subprocess.PIPE, check=True)
else:
    print(info_str + 'Calculation of effective cross-section not attempted')
# time counter end - sig_eff calculation
toc_sig_eff = time.perf_counter()
ttot_sig_eff = round(toc_sig_eff - tic_sig_eff,3)
print(f'Time taken for calculating SAS pattern: {ttot_sig_eff} s')
print('-------------------------------------------------')

# Time satistics
print('')
print('-------------------------------------------------')
print('Time statistics')
print('-------------------------------------------------')
print(f'Mesh generation     :{ttot_sg} s')
print(f'SLD assign          :{ttot_sim} s')
print(f'SAS calculation     :{ttot_scatt} s')
print(f'Sig eff calculation :{ttot_sig_eff} s')
print(f'Total               :{ttot_sg + ttot_sim + ttot_scatt+ttot_sig_eff} s')
print('-------------------------------------------------')
