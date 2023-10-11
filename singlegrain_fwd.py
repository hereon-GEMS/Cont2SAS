#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This is an edition of ballinbox_3d.py.

Change log:
    1) ball in box is not there
    2) only node node, connectivity and cell gen

Created on Tue Oct  3 20:41:41 2023

@author: amajumda
"""
from lib import struct_gen as sg
from lib import plotter as pltr
from lib import simulation as sim
from lib import processing as procs
from lib import datasaver as dsv

import os
import time
import argparse
import sys
import logging
import h5py
import subprocess
import numpy as np

# Create a logger
logger = logging.getLogger('single_fwd_logger')

# Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR)
logger.setLevel(logging.INFO)

# Create a file handler to write log messages to a file
file_handler = logging.FileHandler('single_fwd.log')

# Create a console handler to display log messages in the console
console_handler = logging.StreamHandler()

# Create a formatter to specify the log message format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Set the formatter for the handlers
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

"""
configuration of the simulation box
"""
    
#box side lengths
length_a=100
length_b=100
length_c=100
# number of cells in each direction
nx=50
ny=50
nz=50

# cell volume 
cell_vol=(length_a/nx)*(length_a/ny)*(length_a/nz)
#cell_vol=1

# time steps
num_time_step=11
sld_in=1
sld_out=0

#
mode='shrink'
rad_start_shrnk=35
rad_end_shrnk=5
    

# result folder structure
os.makedirs('data', exist_ok=True)
res_folder=os.path.join('./data/',
                        str(length_a)+'_'+str(length_b)+'_'+str(length_c)+'_'+
                        str(nx)+'_'+str(ny)+'_'+str(ny))


# get nodes, cells, connectivity
data_filename=os.path.join(res_folder,'data.h5')
data_file=h5py.File(data_filename, 'r')
nodes=data_file['nodes'][:]
cells=data_file['cells'][:]
con=data_file['connectivity'][:]
data_file.close()

# create 
dyn_file='single_atom_'+ mode
dyn_folder=os.path.join(res_folder,dyn_file)
os.system('rm -r {0}*'.format(dyn_folder))
simu_save_t1=time.perf_counter()
if mode=='shrink':
    print('creating simulation with schrinking grain')
    print('data will be saved in folder {0}'.format(dyn_folder))
    
neutron_count=np.zeros(num_time_step)
for i in range(num_time_step):
    
    time_dir=os.path.join(dyn_folder,'t{0:0>3}'.format(i))
    if mode=='shrink':
        rad_t=rad_start_shrnk+((rad_end_shrnk-rad_start_shrnk)/(num_time_step-1))*i
        print('radius : {0}'.format(rad_t))        
        simu_t1=time.perf_counter()
        #### simulation start ###
        sld_dyn = sim.sph_grain_3d(nodes,[length_a/2,length_b/2,length_c/2],rad_t,sld_in,sld_out)
        sld_dyn_cell=procs.node2cell_3d_t(nodes , cells, con, sld_dyn, nx, ny, nz, cell_vol)
        sld_dyn_cell_cat, cat = procs.categorize_prop_3d_t(sld_dyn_cell, 10)
        #### simulation end ###
        simu_t2=time.perf_counter()
        print('\t calculating sld distribution took {} S'.format(simu_t2-simu_t1))

    save_t1=time.perf_counter()
    os.makedirs(time_dir, exist_ok=True)
    ### data saving start ###
    data_file_full=os.path.join(time_dir,'data_{0:0>3}.h5'.format(i))
    data_file=h5py.File(data_file_full,'w')
    data_file['node']=nodes
    data_file['nodeprop']=sld_dyn
    data_file['cell']=cells
    data_file['cellprop']=sld_dyn_cell
    data_file['catcellprop']=sld_dyn_cell_cat
    data_file['catcell']=cat
    data_file['mode']=mode
    data_file['radius']=rad_t
    data_file['grain_sld']=sld_in
    data_file['env_sld']=sld_out
    data_file.close()
    ### data saving end ###
    save_t2=time.perf_counter()
    print('\t saving data_{0:0>3}.h5 took {1} S'.format(i, save_t2-save_t1))
    
    
    ### pdb dcd generation ###
    pdb_dcd_dir=os.path.join(time_dir,'pdb_dcd')
    os.makedirs(pdb_dcd_dir, exist_ok=True)
    dsv.pdb_dcd_gen_opt1(cells, sld_dyn_cell_cat, cat, pdb_dcd_dir)
    
    ### scatter.xml generate ###
    dsv.scatterxml_generator(time_dir, sigfile='signal.h5')
    
    ### database generator ###
    db_dir=os.path.join(time_dir,'database')
    dsv.database_generator(min(sld_dyn_cell_cat), max(sld_dyn_cell_cat), ndiv=10, database_dir=db_dir)
    
    ### sassena runner ###
    #os.system('cd ' + time_dir)
    parent_dir=os.getcwd()
    os.chdir(os.path.join(parent_dir,time_dir))
    #print(os.getcwd())
    os.system('mpirun -np 8 sassena')
    os.chdir(parent_dir)
    #print(os.getcwd())
    
    ### 
    sigfile_name=os.path.join(time_dir,'signal.h5')
    sigfile=h5py.File(sigfile_name, 'r')
    q_vec_t=np.sqrt(np.sum(sigfile['qvectors'][:]**2,axis=1))
    q_args=np.argsort(q_vec_t)
    fq0_t=np.sqrt(np.sum(sigfile['fq0'][:]**2,axis=1))
    q_vec_t=q_vec_t[q_args]
    fq0_t=fq0_t[q_args]
    sigfile.close()
    neutron_count_t=0
    for j in range(len(q_vec_t)-1):
        del_q=q_vec_t[j+1]-q_vec_t[j]
        neutron_count_t+=0.5*del_q*(fq0_t[j+1]+fq0_t[j])
    print('neutron count is '+ str(neutron_count_t))
    neutron_count[i]=neutron_count_t
#countdatadir=os.path.join(res_folder, dyn_file)
countdatafile_name=os.path.join(dyn_folder, 'count.h5')
countdatafile=h5py.File(countdatafile_name, 'w')
countdatafile['count']=neutron_count
countdatafile['time']=np.linspace(0,num_time_step-1,num_time_step)
countdatafile.close()
simu_save_t2=time.perf_counter()
print('Total time taken is {0} S'.format(simu_save_t2-simu_save_t1))




