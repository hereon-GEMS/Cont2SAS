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
# mid point of structure
mid_point=np.array([length_a/2, length_b/2, length_c/2])
# element type
el_type=root.find('element').find('type').text
if el_type=='lagrangian':
    el_order=int(root.find('element').find('order').text)
    el_info={'type': el_type,
             'order': el_order}

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
# sim_dir=os.path.join('../data/',
#                         length_a_str+'_'+length_b_str+'_'+length_c_str+'_'+
#                         nx_str+'_'+ny_str+'_'+nz_str+'/simulation')
os.makedirs(sim_dir, exist_ok=True)

# read structure info
data_filename=os.path.join(sim_dir,'../structure/struct.h5')
nodes, cells, con = dsv.mesh_read(data_filename)

# folder name for model
model_dir_name= (sim_model + '_tend_' + str(t_end) + '_dt_' + str(dt) \
    + '_ensem_' + str(n_ensem)).replace('.','p')
model_dir=os.path.join(sim_dir,model_dir_name)

# folder name for model with particular run param
model_xml=os.path.join(xml_folder, 'model_'+sim_model + '.xml')
tree = ET.parse(model_xml)
root = tree.getroot()
model_param_dir_name=''
for elem in root.iter():
    if elem.text and elem.text.strip():  # Avoid None or empty texts
        model_param_dir_name+= f"{elem.tag}_{elem.text.strip()}_"
model_param_dir_name=model_param_dir_name[0:-1].replace('.', 'p')
model_param_dir=os.path.join(model_dir,model_param_dir_name)

if os.path.exists(model_param_dir):
    print('calculating scattering function')
else:
    print('create simulation first')

fit_param_arr_1=np.zeros(len(t_arr))
fit_param_arr_2=np.zeros(len(t_arr))
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
    #print(Iq_num)
    
    tree=ET.parse(model_xml)
    root = tree.getroot()

    # decreitization params
    # number of categories and method of categorization
    # ball_sld=float(root.find('sld_in').text)
    # box_sld=float(root.find('sld_out').text)
    # r_0=float(root.find('rad_0').text)
    # r_end=float(root.find('rad_end').text)
    # ball_rad=r_0+((r_end-r_0)/t_end)*t
    # fit_param=fit.fitter(Iq_num=Iq_num, q_num=q_num, 
    #                         model_xml=model_xml, 
    #                         t_end=t_end, t=t)
    ball_sld=float(root.find('sld_in').text)
    box_sld=float(root.find('sld_out').text)
    ball_rad=float(root.find('rad').text)
    sig_0=float(root.find('sig_0').text)
    sig_end=float(root.find('sig_end').text)
    # Iq_fit, q_fit =ana.ball(qmax=np.max(q_num),qmin=np.min(q_num),Npts=100,
    #             scale=1,bg=0,sld=ball_sld,sld_sol=box_sld,rad=fit_param)
    fit_param=sig_0+((sig_end-sig_0)/t_end)*t
    fit_param_arr_1, fit_param_arr_2=fit.fitter(Iq_num=Iq_num, q_num=q_num, 
                         model_xml=model_xml, 
                         t_arr=t_arr, t_idx=i, 
                         fit_param_1=fit_param_arr_1, fit_param_2=fit_param_arr_2)
    # print(fit_param)
    Iq_fit, q_fit =ana.fuzzysph(qmax=np.max(q_num),qmin=np.min(q_num),Npts=100,scale=1,
                                bg=0,sld=ball_sld,sld_sol=box_sld,
                                sig_fuzz=fit_param_arr_1[i], radius=fit_param_arr_2[i])
    
    
    vol_box=length_a*length_b*length_c
    # vol_ball=(4/3)*np.pi*ball_rad**3
    plt.loglog(q_num, Iq_num,'o')
    plt.loglog(q_fit, Iq_fit)
    plt.show()

    # fit_param_arr[i] =  fit_param

# # fit param saving
# fit_param_dir = model_param_dir
# fit_param_file_name = 'fit_param.h5'
# fit_param_file = os.path.join(fit_param_dir, 'fit_param.h5')
# fit_param_name = fit.get_fit_param(sim_model)
# fit_param_data=h5py.File(fit_param_file,'w')
# fit_param_data['fit_param_val']=fit_param_arr
# fit_param_data['fit_param_name']=fit_param_name
# fit_param_data.close()




    # Iq_all_ensem=np.zeros((num_points,n_ensem))
    # q_all_ensem=np.zeros((num_points,n_ensem))
    # for j in range(n_ensem):
    #     idx_ensem=j
    #     # create ensemble dir
    #     ensem_dir_name='ensem{0:0>3}'.format(idx_ensem)
    #     ensem_dir=os.path.join(t_dir, ensem_dir_name)
    #     """
    #     discretization
    #     """
    #     # read node sld
    #     sim_data_file_name='sim.h5'
    #     sim_data_file=os.path.join(ensem_dir, sim_data_file_name)
    #     sim_data=h5py.File(sim_data_file,'r')
    #     node_sld=sim_data['sld'][:]
    #     sim_data.close()

    #     # create scatt dir
    #     scatt_dir_name='scatt_cal_'+ scatt_settings
    #     scatt_dir=os.path.join(ensem_dir, scatt_dir_name)
    #     os.makedirs(scatt_dir, exist_ok=True)
    #     # calculate pseudo atom position
    #     pseudo_pos=cells


    #     # calculate pseudo atom scattering lengths
    #     cell_dx=length_a/nx
    #     cell_dy=length_b/ny
    #     cell_dz=length_b/nz
    #     cell_vol=cell_dx*cell_dy*cell_dz
    #     pseudo_b=scatt.pseudo_b(node_sld,con,cell_vol)
    #     #sld_dyn_cell_cat, cat = procs.categorize_prop_3d_t(sld_dyn_cell, 10)

    #     # categorize SLD
    #     pseudo_b_cat_val, pseudo_b_cat_idx = scatt.pseudo_b_cat(pseudo_b,num_cat,method=method_cat)

    #     # save pseudo atom info
    #     scatt_cal_data_file_name='scatt_cal.h5'
    #     scatt_cal_data_file=os.path.join(scatt_dir, scatt_cal_data_file_name)
    #     scatt_cal_data=h5py.File(scatt_cal_data_file,'w')
    #     scatt_cal_data['node_pos']=nodes
    #     scatt_cal_data['node_sld']=node_sld
    #     scatt_cal_data['pseudo_pos']=pseudo_pos
    #     scatt_cal_data['pseudo_b']=pseudo_b
    #     scatt_cal_data['pseudo_b_cat_val']=pseudo_b_cat_val
    #     scatt_cal_data['pseudo_b_cat_idx']=pseudo_b_cat_idx
    #     scatt_cal_data.close()

    #     """
    #     calculate I vs Q
    #     """
        
    #     ### pdb dcd generation ###
    #     pdb_dcd_dir=os.path.join(scatt_dir,'pdb_dcd')
    #     os.makedirs(pdb_dcd_dir, exist_ok=True)
    #     scatt.pdb_dcd_gen(pdb_dcd_dir, pseudo_pos, pseudo_b_cat_val, pseudo_b_cat_idx)

    #     ### database generator ###
    #     db_dir_name='database'
    #     db_dir=os.path.join(scatt_dir,db_dir_name)
    #     os.makedirs(db_dir, exist_ok=True)
    #     scatt.db_gen(db_dir, pseudo_b_cat_val, pseudo_b_cat_idx)
        
    #     ### scatter.xml generate ###
    #     # original command 
    #     # dsv.scatterxml_generator(time_dir, sigfile='signal.h5')
    #     # detailed version
    #     scatter_xml_file_name='scatter.xml'
    #     scatter_xml_file=os.path.join(scatt_dir,scatter_xml_file_name)
    #     qclean_sld=scatt.qclean_sld(sim_model, xml_folder)
    #     scatt.scattxml_gen(scatter_xml_file, signal_file,scan_vector,
    #                         start_length, end_length, num_points,
    #                           resolution_num, qclean_sld, length_a, length_b, length_c, mid_point)

    #     # ### sassena runner ###
    #     parent_dir=os.getcwd()
    #     os.chdir(os.path.join(parent_dir,scatt_dir))
    #     sassena_exec='/home/amajumda/Documents/Softwares/sassena/compile/sassena'
    #     sass_out_file='sass.log'
    #     if os.path.exists(signal_file):
    #         os.remove(signal_file)
    #     os.system('mpirun -np 8 {0} > {1} 2>&1'.format(sassena_exec, sass_out_file))
    #     os.chdir(parent_dir)

    #     # read and save Iq data from current ensem
    #     ## read
    #     signal_file_loc=os.path.join(scatt_dir, signal_file)
    #     sig_data=h5py.File(signal_file_loc,'r')
    #     Iq_ensem=np.sqrt(np.sum(sig_data['fq'][:]**2,axis=1))
    #     q_ensem=np.sqrt(np.sum(sig_data['qvectors'][:]**2,axis=1))
    #     sig_data.close()
    #     # process
    #     q_arg_ensem=np.argsort(q_ensem)
    #     Iq_ensem=Iq_ensem[q_arg_ensem]
    #     q_ensem=q_ensem[q_arg_ensem]
    #     ## save
    #     Iq_all_ensem[:,idx_ensem]=Iq_ensem
    #     q_all_ensem[:,idx_ensem]=q_ensem

    # # ensem average
    # Iq=np.average(Iq_all_ensem, axis=1)
    # q=q_all_ensem[:,0]
    # q_arg=np.argsort(q)
    # Iq=Iq[q_arg]
    # q=q[q_arg]
    # # plotitng I vs Q in time folder
    # plt.loglog(q,Iq)
    # plt.xlabel('Q')
    # plt.ylabel('I(Q)')
    # Iq_plot_file_name='Iq_{0}.jpg'.format(scatt_settings) 
    # Iq_plot_file=os.path.join(t_dir, Iq_plot_file_name)
    # plt.savefig(Iq_plot_file, format='jpg')
    # plt.show()
    # # saving I vs Q in time folder
    # Iq_data_file_name='Iq_{0}.h5'.format(scatt_settings) 
    # Iq_data_file=os.path.join(t_dir,Iq_data_file_name)
    # Iq_data=h5py.File(Iq_data_file,'w')
    # Iq_data['Iq']=Iq
    # Iq_data['Q']=q
    # Iq_data.close()