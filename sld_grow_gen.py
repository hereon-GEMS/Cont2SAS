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

### sim_gen ###
sim_model='sld_grow'
dt=1 
t_end=10
n_ensem=1

### model_param ###
rad=10
sld_in_0=2
sld_in_end=5
sld_out=1

### scatt_cal ###
num_cat=51
method_cat='extend'
sassena_exe= '/home/amajumda/Documents/Softwares/sassena/compile/sassena'
mpi_procs=2
num_threads=2
sig_file='signal.h5'
scan_vec=np.array([1, 0, 0])
Q_range=np.array([0.0029, 0.05])
num_points=100
num_orientation=100

### sig_eff_cal ###
# distance : distance between detenctor and sample
# wl : neutron wave length
# beam_center_coord : beam center off set from detector center
instrument='SANS-1'
facility='MLZ'
distance=20
wl=4.5
beam_center_coord=np.array([0, 0, 0])

### run_vars ###
struct_gen_run=0
sim_run=0
scatt_cal_run=1
sig_eff_cal_run=1

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
    print(info_str + 'Generating model_sld_grow.xml')
    #xml_gen.model_ball_xml_write(xml_dir, rad, sld)
    xml_gen.model_sld_grow_xml_write(xml_dir, rad, sld_in_0, sld_in_end, sld_out)

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

print(part_str + 'Calculating effective cross-section')
if sig_eff_cal_run==1:
    # scatt_cal xml
    print(info_str + 'Generating sig_eff_cal.xml')
    xml_gen.sig_eff_xml_write(xml_dir, instrument, facility, 
                        distance, wl, beam_center_coord)

    # calculate scattering function
    print(info_str + 'Executing sig_eff.py')
    script_path = os.path.join(script_dir, 'sig_eff_cal.py')  # Full path of the script
    #working_dir = "."  # Directory to run the script in
    subprocess.run(["python", script_path], cwd=working_dir, stderr=subprocess.PIPE)
else:
    print(info_str + 'Calculation of effective cross-section not attempted')
