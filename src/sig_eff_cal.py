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
# mid point of structure
mid_point=np.array([length_a/2, length_b/2, length_c/2])

### sim xml ###

sim_xml=os.path.join(xml_folder, 'simulation.xml')

tree=ET.parse(sim_xml)
root = tree.getroot()

# model name
sim_model=root.find('model').text

# simulation parameters
## time
dt=float(root.find('sim_param').find('dt').text)
t_end=float(root.find('sim_param').find('tend').text)
t_arr=np.arange(0,t_end+dt, dt)
## ensemble
n_ensem=int(root.find('sim_param').find('n_ensem').text)

# scatter calculation
# scatt_cal
scatt_cal_xml=os.path.join(xml_folder, 'scatt_cal.xml')

tree=ET.parse(scatt_cal_xml)
root = tree.getroot()

# decreitization params
# number of categories and method of categorization
num_cat=int(root.find('discretization').find('num_cat').text)
method_cat=root.find('discretization').find('method_cat').text

# scatt_cal params
signal_file=root.find('scatt_cal').find('sig_file').text
resolution_num=int(root.find('scatt_cal').find('num_orientation').text)
start_length=float(root.find('scatt_cal').find('Q_start').text)
end_length=float(root.find('scatt_cal').find('Q_end').text)
num_points=int(root.find('scatt_cal').find('num_points').text)
scan_vec_x=float(root.find('scatt_cal').find('scan_vec').find('x').text)
scan_vec_y=float(root.find('scatt_cal').find('scan_vec').find('y').text)
scan_vec_z=float(root.find('scatt_cal').find('scan_vec').find('z').text)
scan_vector=[scan_vec_x, scan_vec_y, scan_vec_z]


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
sim_dir=os.path.join('../data/',
                        length_a_str+'_'+length_b_str+'_'+length_c_str+'_'+
                        nx_str+'_'+ny_str+'_'+nz_str+'/simulation')
os.makedirs(sim_dir, exist_ok=True)

# read structure info
data_filename=os.path.join(sim_dir,'../structure/struct.h5')
nodes, cells, con = dsv.mesh_read(data_filename)

# folder name for model
model_dir_name= (sim_model + '_tend_' + str(t_end) + '_dt_' + str(dt) \
    + '_ensem_' + str(n_ensem)).replace('.','p')
model_dir=os.path.join(sim_dir,model_dir_name)

# folder name for run of model with particular run param
model_xml=os.path.join(xml_folder, 'model_'+sim_model + '.xml')
tree = ET.parse(model_xml)
root = tree.getroot()
model_param_dir_name=''
for elem in root.iter():
    if elem.text and elem.text.strip():  # Avoid None or empty texts
        model_param_dir_name+= f"{elem.tag}_{elem.text.strip()}_"
model_param_dir_name=model_param_dir_name[0:-1]
model_param_dir=os.path.join(model_dir,model_param_dir_name)

if os.path.exists(model_param_dir):
    print('calculating effective cross-section')
else:
    print('create simulation first')

for i in range(len(t_arr)):
    t=t_arr[i]
    # time_dir name
    t_dir_name='t{0:0>3}'.format(i)
    t_dir=os.path.join(model_param_dir, t_dir_name)
    Iq_all_ensem=np.zeros((num_points,n_ensem))
    q_all_ensem=np.zeros((num_points,n_ensem))
    # read I and Q
    Iq_data_file_name='Iq.h5'
    Iq_data_file=os.path.join(t_dir,Iq_data_file_name)
    Iq_data=h5py.File(Iq_data_file,'r')
    Iq=Iq_data['Iq'][:]
    q=Iq_data['Q'][:]
    Iq_data.close()
    # plotitng I vs Q in time folder
    plt.loglog(q,Iq)
    plt.xlabel('Q')
    plt.ylabel('I(Q)')
    plt.show()
    # calculate sigma eff
    sig_eff=np.zeros(len(t_arr))
    #neutron_count_t=0
    for j in range(len(q)-1):
        del_q=q[j+1]-q[j]
        sig_eff[i]+=0.5*del_q*(Iq[j+1]+Iq[j])
    # neutron_count[i]=neutron_count_t
sig_eff_data_file_name='sig_eff.h5'
sig_eff_data_file=os.path.join(model_param_dir,sig_eff_data_file_name)
sig_eff_data=h5py.File(sig_eff_data_file,'w')
sig_eff_data['sig_eff']=sig_eff
sig_eff_data['t']=t_arr
sig_eff_data.close()

plt.plot(t_arr, sig_eff)
plt.show()