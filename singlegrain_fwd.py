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
# from lib import struct_gen as sg
# from lib import plotter as pltr
from lib import simulation as sim
from lib import processing as procs
from lib import datasaver as dsv

import os
import time
# import argparse
# import sys
# import logging
import h5py
# import subprocess
import numpy as np


"""
configuration of the simulation box
"""
    
#box side lengths
length_a=20
length_b=20
length_c=20
# number of cells in each direction
nx=40
ny=40
nz=40

# cell volume 
cell_vol=(length_a/nx)*(length_a/ny)*(length_a/nz)

# time steps
num_time_step=11
dt=0.5
sld_in=1
sld_out=0

#
mode='diffuse'


if mode=='shrink':
    rad_start_shrnk=35
    rad_end_shrnk=5

if mode=='diffuse':
    rad=35
    D=0.07

# if mode=='gg':
#     rad=3
#     D=0.07
    

# result folder structure
os.makedirs('data', exist_ok=True)
res_folder=os.path.join('./data/',
                        str(length_a)+'_'+str(length_b)+'_'+str(length_c)+'_'+
                        str(nx)+'_'+str(ny)+'_'+str(ny))


# get nodes, cells, connectivity
data_filename=os.path.join(res_folder,'data.h5')
nodes, cells, con = dsv.mesh_read(data_filename)

# create dolder for dynamics
dyn_file='single_grain_'+ mode
dyn_folder=os.path.join(res_folder,dyn_file)
os.makedirs(dyn_folder, exist_ok=True)
os.system('rm -r {0}/*'.format(dyn_folder))
simu_save_t1=time.perf_counter() # time counter for simulation

if mode=='shrink':
    print('creating simulation with schrinking grain')

if mode== 'diffuse':
    print('creating simulation with diffusion into grain')

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
    if mode=='diffuse':
        #rad_t=rad_start_shrnk+((rad_end_shrnk-rad_start_shrnk)/(num_time_step-1))*i
        print('timestep : {0}'.format(i))        
        simu_t1=time.perf_counter()
        #### simulation start ###
        #sld_dyn = sim.sph_grain_3d(nodes,[length_a/2,length_b/2,length_c/2],rad_t,sld_in,sld_out)
        sld_dyn = sim.sph_grain_diffus_book_1_3d(nodes,
                                                 [length_a/2,length_b/2,length_c/2],
                                                 rad, D, dt*i, sld_in,sld_out)
        sld_dyn_cell=procs.node2cell_3d_t(nodes , cells, con, sld_dyn, nx, ny, nz, cell_vol)
        sld_dyn_cell_cat, cat = procs.categorize_prop_3d_t(sld_dyn_cell, 10)
        #### simulation end ###
        simu_t2=time.perf_counter()
        print('\t calculating sld distribution took {} S'.format(simu_t2-simu_t1))

    save_t1=time.perf_counter()
    os.makedirs(time_dir, exist_ok=True)
    ### data saving start ###
    data_file_full=os.path.join(time_dir,'data_{0:0>3}.h5'.format(i))
    if mode=='shrink':
        dsv.sim_gen(data_file_full, nodes, sld_dyn, cells, sld_dyn_cell, sld_dyn_cell_cat,
            cat, mode, sld_in, sld_out, (rad_t))
    if mode=='diffuse':
        dsv.sim_gen(data_file_full, nodes, sld_dyn, cells, sld_dyn_cell, sld_dyn_cell_cat,
            cat, mode, sld_in, sld_out, (rad, D))
   
    # ### data saving end ###
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
    parent_dir=os.getcwd()
    os.chdir(os.path.join(parent_dir,time_dir))
    os.system('mpirun -np 8 sassena')
    os.chdir(parent_dir)
    
    ### 
    sigfile_name=os.path.join(time_dir,'signal.h5')
    q_vec_t, fq0_t, fqt_t, fq_t, fq2_t = dsv.sig_read(sigfile_name)
    neutron_count_t=0
    for j in range(len(q_vec_t)-1):
        del_q=q_vec_t[j+1]-q_vec_t[j]
        neutron_count_t+=0.5*del_q*(fq0_t[j+1]+fq0_t[j])
    print('neutron count is '+ str(neutron_count_t))
    neutron_count[i]=neutron_count_t
#countdatadir=os.path.join(res_folder, dyn_file)
countdatafile_name=os.path.join(dyn_folder, 'count.h5')
time_arr= np.linspace(0,num_time_step-1,num_time_step)
dsv.count_gen(countdatafile_name, neutron_count, time_arr)

simu_save_t2=time.perf_counter()
print('Total time taken is {0} S'.format(simu_save_t2-simu_save_t1))

