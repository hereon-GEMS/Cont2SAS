"""
This generates data for phase field model
simulated using moose
data is saved in data folder

Author: Arnab Majumdar
Date: 24.06.2025
"""
from lib import xml_gen
import subprocess
import numpy as np
import os
import h5py
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

# phase field simulation details 
phenm='spinodal_fe_cr'
sim_times=(np.array([0, 2, 4, 6, 8, 8.64]) * 10000).astype(int)
# sim_times=(np.array([8.64]) * 10000).astype(int) # uncomment when using 3 categories

# initialize time vars
ttot_sg_sum=0
ttot_sim_sum=0
ttot_scatt_sum=0
ttot_sum=0

for t_idx, sim_time in enumerate(sim_times):
    print(f'-------------------------------------------------')
    print(f'/////////////////////////////////////////////////')
    print(f'Simulation time: {sim_time}')
    print(f'/////////////////////////////////////////////////')
    print(f'-------------------------------------------------')
    # moose input file
    moose_inp_file_name='{0}_{1:0>5}.h5'.format(phenm, sim_time)
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
    sim_time=moose_inp['time'][()]
    qclean_sld=moose_inp['qclean_sld'][()]

    ### scatt_cal ###
    num_cat=501 # use 3 for the categorization picture
    method_cat='extend'
    sassena_exe= './Sassena/Sassena.AppImage'
    mpi_procs=4
    num_threads=2
    sig_file='signal.h5'
    scan_vec=np.array([1, 0, 0])
    Q_range=np.array([2*np.pi/length_a, 1])
    num_points=100
    num_orientation=200

    ### run_vars ###
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
        print(info_str + f'Generating model_{sim_model}.xml')
        # xml_gen.model_ball_xml_write(xml_dir, rad, sld)
        # xml_gen.model_fs_xml_write(xml_dir, rad, sig_0, sig_end, sld_in, sld_out)
        # print(time)
        xml_gen.model_phase_field_xml_write(xml_dir, name, sim_time, qclean_sld)

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
    moose_inp.close()
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

    # total time taken
    ttot_sg_sum+=ttot_sg
    ttot_sim_sum+=ttot_sim
    ttot_scatt_sum+=ttot_scatt
    ttot_sum+=ttot_sg + ttot_sim + ttot_scatt

# Time satistics for all
print('')
print(f'-------------------------------------------------')
print(f'Time statistics')
print(f'-------------------------------------------------')
print(f'Mesh generation :{ttot_sg_sum} s')
print(f'SLD assign      :{ttot_sim_sum} s')
print(f'SAS calculation :{ttot_scatt_sum} s')
print(f'Total           :{ttot_sum} s')
print(f'-------------------------------------------------')