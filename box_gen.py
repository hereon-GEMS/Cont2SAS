"""
This generates data for box model
data is saved in data folder

Author: Arnab Majumdar
Date: 24.06.2025
"""

from lib import xml_gen
import subprocess
import numpy as np
import os
import time

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

"""
input values
"""
# script dir and working dir
script_dir = "src"  # Full path of the script
working_dir = "."  # Directory to run the script in


### struct gen ###
xml_dir='./xml' 
length_a=40.
length_b=40. 
length_c=40.
nx=40 
ny=40 
nz=40 
el_type='lagrangian'
el_order=1
update_val=True
plt_node=False
plt_cell=False
plt_mesh=False

### sim gen ###
sim_model='box'
dt=1
t_end=0
n_ensem=1

### model param ###
sld=2
qclean_sld=0

### scatt cal ###
num_cat=3
method_cat='extend'
sassena_exe= '/home/amajumda/Documents/Softwares/sassena/compile/sassena'
mpi_procs=4
num_threads=2
sig_file='signal.h5'
scan_vec=np.array([1, 0, 0])
Q_range=np.array([0, 1])
num_points=100
num_orientation=10

### run vars ###
struct_gen_run=1
sim_run=1
scatt_cal_run=1

"""
xml gen and run scripts
"""
part_str='Part>>>  '
info_str='Info>>>  '

print(f'-------------------------------------------------')
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
    subprocess.run(["python", script_path], cwd=working_dir, stderr=subprocess.PIPE)
else:
    print(info_str + 'structure generation not attempted')
# time counter end - struct gen 
toc_sg = time.perf_counter()
ttot_sg= round(toc_sg - tic_sg,3)
print(f'Time taken for generating mesh: {ttot_sg} s')
print(f'-------------------------------------------------')

print(f'-------------------------------------------------')
print(part_str + 'Assigning SLD values to nodes')
print('')
# time counter start - simulation gen
tic_sim = time.perf_counter()
if sim_run==1:
    # simulation xml
    print(info_str + 'Generating simulation.xml')
    xml_gen.sim_xml_write(xml_dir, sim_model,dt, t_end, n_ensem)

    # model xml
    print(info_str + 'Generating model_ball.xml')
    xml_gen.model_box_xml_write(xml_dir, sld, qclean_sld)

    # generate structure
    print(info_str + 'Executing sim_gen.py')
    script_path = os.path.join(script_dir, 'sim_gen.py')  # Full path of the script
    #working_dir = "."  # Directory to run the script in
    subprocess.run(["python", script_path], cwd=working_dir, stderr=subprocess.PIPE)
else:
    print(info_str + 'simulation not attempted')
# time counter end - simulation gen 
toc_sim = time.perf_counter()
ttot_sim= round(toc_sim - tic_sim,3)
print(f'Time taken for assigning SLD values: {ttot_sim} s')
print(f'-------------------------------------------------')

print(f'-------------------------------------------------')
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
    subprocess.run(["python", script_path], cwd=working_dir, stderr=subprocess.PIPE)
else:
    print(info_str + 'Calculation of scattring function not attempted')
# time counter end - SAS calculation
toc_scatt = time.perf_counter()
ttot_scatt= round(toc_scatt - tic_scatt,3)
print(f'Time taken for calculating SAS pattern: {ttot_sim} s')
print(f'-------------------------------------------------')


# Time satistics
print('')
print(f'-------------------------------------------------')
print(f'Time statistics')
print(f'-------------------------------------------------')
print(f'Mesh generation :{ttot_sg} s')
print(f'SLD assign      :{ttot_sim} s')
print(f'SAS calculation :{ttot_scatt} s')
print(f'Total           :{ttot_sg + ttot_sim + ttot_scatt} s')
print(f'-------------------------------------------------')