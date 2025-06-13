from lib import xml_gen
import subprocess
import numpy as np
import os
import h5py

"""
input values
"""
# script dir and working dir
script_dir = "src"  # Full path of the script
working_dir = "."  # Directory to run the script in

# phase field simulation details 
phenm='spinodal_fe_cr'
times=(np.array([0, 2, 4, 6, 8, 8.64]) * 10000).astype(int)
# times=(np.array([0]) * 10000).astype(int)

for t_idx, time in enumerate(times):
    print(time)
    moose_inp_file_name='{0}_{1:0>5}.h5'.format(phenm, time)
    moose_inp_file=os.path.join('moose', moose_inp_file_name)
    moose_inp=h5py.File(moose_inp_file,'r')

    ### struct gen ###
    xml_dir='./xml' 
    length_a=moose_inp['length_a'][()]
    length_b=moose_inp['length_b'][()]
    length_c=moose_inp['length_c'][()]
    nx=moose_inp['nx'][()]
    ny=moose_inp['ny'][()]
    nz=moose_inp['ny'][()]
    el_type='lagrangian'
    el_order=1
    update_val=True
    plt_node=False
    plt_cell=False
    plt_mesh=False

    ### sim_gen ###
    sim_model='phase_field'
    dt=1 
    t_end=0
    n_ensem=1

    ### model_param ###
    name=phenm
    time=moose_inp['time'][()]
    qclean_sld=moose_inp['qclean_sld'][()]

    ### scatt_cal ###
    num_cat=501
    method_cat='extend'
    sassena_exe= '/home/amajumda/Documents/Softwares/sassena/compile/sassena'
    mpi_procs=4
    num_threads=1
    sig_file='signal.h5'
    scan_vec=np.array([1, 0, 0])
    Q_range=np.array([2*np.pi/250, 1])
    num_points=100
    num_orientation=200

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
        print(info_str + f'Generating model_{sim_model}.xml')
        # xml_gen.model_ball_xml_write(xml_dir, rad, sld)
        # xml_gen.model_fs_xml_write(xml_dir, rad, sig_0, sig_end, sld_in, sld_out)
        print(time)
        xml_gen.model_phase_field_xml_write(xml_dir, name, time, qclean_sld)

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
    moose_inp.close()

"""
### struct gen ###
xml_dir='./xml' 
length_a=250.
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
sim_model='phase_field'.format(0)
dt=1 
t_end=0
n_ensem=1

### model_param ###
name='spinodal_fe_cr'
time=0

### scatt_cal ###
num_cat=501
method_cat='extend'
sassena_exe= '/home/amajumda/Documents/Softwares/sassena/compile/sassena'
mpi_procs=6
num_threads=2
sig_file='signal.h5'
scan_vec=np.array([1, 0, 0])
Q_range=np.array([2*np.pi/250, 1])
num_points=100
num_orientation=500

### run_vars ###
struct_gen_run=1
sim_run=1
scatt_cal_run=1

"""
#xml gen and run scripts
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
    # print(info_str + 'Generating model_fs.xml')
    #xml_gen.model_ball_xml_write(xml_dir, rad, sld)
    # xml_gen.model_fs_xml_write(xml_dir, rad, sig_0, sig_end, sld_in, sld_out)

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
"""