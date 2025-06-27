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
from lib import scatt_cal as scatt
from lib import fitter as fit
from lib import analytical as ana



import os
import time
import argparse
import sys
import xml.etree.ElementTree as ET
import numpy as np
import subprocess
import matplotlib.pyplot as plt
import h5py
import imageio.v2 as imageio
import mdtraj as md



#timer counter initial
tic = time.perf_counter()

"""
read input from xml file
"""

### struct xml ###

# box side lengths
length_a=40.
length_b=40.
length_c=40.
# number of cells in each direction
nx=160
ny=nx
nz=nx
# mid point of structure
mid_point=np.array([length_a/2, length_b/2, length_c/2])
# element type
el_type='lagrangian'
el_order=2

### sim xml ###

# model name
sim_model='fs'

# simulation parameters
## time
dt=10.
t_end=10.
t_arr=np.arange(0,t_end+dt, dt)
## ensemble
n_ensem=1

# model params
rad=10
sig_0=0
sig_end=4
sld_in=2
sld_out=0

# scatter calculation
# scatt_cal xml

# decreitization params
# number of categories and method of categorization
num_cat=101
method_cat='extend'
#sassena
sassena_exe= '/home/amajumda/Documents/Softwares/sassena/compile/sassena'
mpi_procs=4
num_threads=2
# scatt_cal params
signal_file='signal.h5'
resolution_num=10
start_length=0.
end_length=1.
num_points=100
scan_vec_x=0
scan_vec_y=0
scan_vec_z=1
scan_vector=[scan_vec_x, scan_vec_y, scan_vec_z]
# signal_file=root.find('scatt_cal').find('sig_file').text
scatt_settings='cat_' + method_cat + '_' + str(num_cat) + 'Q_' \
    + str(start_length) + '_' + str(end_length) + '_' + 'orien_' + '_' + str(resolution_num)
scatt_settings=scatt_settings.replace('.', 'p')

"""
create folder structure, read structure and sld info
"""

# folder structure
## mother folder for simulation 
### save length values as strings
### decimal points are replaced with p
length_a_str=str(length_a).replace('.','p')
length_b_str=str(length_a).replace('.','p')
length_c_str=str(length_a).replace('.','p')

### save num_cell values as strings
nx_str=str(nx)
ny_str=str(ny)
nz_str=str(nz)
struct_folder_name = (length_a_str + '_' + length_b_str + '_' + length_c_str
                       + '_' + nx_str + '_' + ny_str + '_' + nz_str)
# save elemnt type as string
if el_type=='lagrangian':
    el_order_str=str(el_order)
    struct_folder_name += '_' + el_type + '_' + el_order_str


sim_dir=os.path.join('./data/', struct_folder_name +'/simulation')
os.makedirs(sim_dir, exist_ok=True)

# read structure info
data_filename=os.path.join(sim_dir,'../structure/struct.h5')
nodes, cells, con = dsv.mesh_read(data_filename)

# folder name for model
model_dir_name= (sim_model + '_tend_' + str(t_end) + '_dt_' + str(dt) \
    + '_ensem_' + str(n_ensem)).replace('.','p')
model_dir=os.path.join(sim_dir,model_dir_name)

# folder name for model with particular run param
model_param_dir_name=('rad_' + str(rad) + '_sig_0_' + str(sig_0)
                      + '_sig_end_' + str(sig_end) + '_sld_in_' + str(sld_in)
                      + '_sld_out_' + str(sld_out) + '_')
print(str(sld_out))
model_param_dir_name=model_param_dir_name[0:-1].replace('.', 'p')
model_param_dir=os.path.join(model_dir,model_param_dir_name)

if os.path.exists(model_param_dir):
    print('calculating scattering function')
else:
    print('create simulation first')

fit_param_arr_1=np.zeros(len(t_arr))
fit_param_arr_2=np.zeros(len(t_arr))
print('mesh {0} - {1}'.format(length_a, nx))
print('sigma evolving from {0} to {1}'.format(sig_0, sig_end))
for i in range(len(t_arr)):
    t=t_arr[i]
    # time_dir name
    t_dir_name='t{0:0>3}'.format(i)
    t_dir=os.path.join(model_param_dir, t_dir_name)

    # read I vs Q in time folder
    Iq_data_file_name='Iq_{0}.h5'.format(scatt_settings) 
    Iq_data_file=os.path.join(t_dir,Iq_data_file_name)
    Iq_data=h5py.File(Iq_data_file,'r')
    Iq_num=Iq_data['Iq'][:]
    q_num=Iq_data['Q'][:]
    Iq_data.close()

    ###
    sph_vol=(4/3)*np.pi*rad**3
    del_rho=(sld_in-sld_out)
    Iq0_ana=(sph_vol*del_rho)**2
    Iq0_num=Iq_num[0]
    print('[t = {0}s] Analytical value: {1}, Numerical value: {2}'.format(t,Iq0_ana, Iq0_num))