from lib import xml_gen
import subprocess
import numpy as np
import os

"""
input values
"""
# script dir and working dir
script_dir = "src"  # Full path of the script
working_dir = "."  # Directory to run the script in


### struct gen ###
xml_dir='./xml' 
length_a=200.
length_b=length_a
length_c=length_a
nx=100 
ny=nx 
nz=nx 
el_type='lagrangian'
el_order=1
update_val=True
plt_node=False
plt_cell=False
plt_mesh=False

### sim_gen ###
sim_model='fs'
dt=1 
t_end=10
n_ensem=1

### model_param ###
rad=60
sig_0=2
sig_end=10
sld_in=5
sld_out=1

### scatt_cal ###
num_cat=501
method_cat='extend'
sassena_exe= '/home/amajumda/Documents/Softwares/sassena/compile/sassena'
mpi_procs=2
num_threads=2
sig_file='signal.h5'
scan_vec=np.array([1, 0, 0])
Q_range=np.array([0, 0.2])
num_points=100
num_orientation=100

### run_vars ###
struct_gen_run=1
sim_run=1
scatt_cal_run=1

"""
xml gen and run scripts
"""
part_str='Part>>>'
info_str='Info>>>'

print(part_str + 'Structure generation')
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

print(part_str + 'Simulating')
if sim_run==1:
    # simulation xml
    print(info_str + 'Generating simulation.xml')
    xml_gen.sim_xml_write(xml_dir, sim_model,dt, t_end, n_ensem)

    # model xml
    print(info_str + 'Generating model_fs.xml')
    #xml_gen.model_ball_xml_write(xml_dir, rad, sld)
    xml_gen.model_fs_xml_write(xml_dir, rad, sig_0, sig_end, sld_in, sld_out)

    # generate structure
    print(info_str + 'Executing sim_gen.py')
    script_path = os.path.join(script_dir, 'sim_gen.py')  # Full path of the script
    #working_dir = "."  # Directory to run the script in
    subprocess.run(["python", script_path], cwd=working_dir, stderr=subprocess.PIPE)
else:
    print(info_str + 'simulation not attempted')

print(part_str + 'Calculating scattering function')
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