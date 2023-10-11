#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 10:39:37 2023

@author: amajumda
"""
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
logger = logging.getLogger('multi_fwd_logger')

# Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR)
logger.setLevel(logging.INFO)

# Create a file handler to write log messages to a file
file_handler = logging.FileHandler('multi_fwd.log')

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
length_a=650
length_b=650
length_c=650
# number of cells in each direction
nx=325
ny=325
nz=325

# cell volume 
cell_vol=(length_a/nx)*(length_a/ny)*(length_a/nz)
#cell_vol=1

# time steps
num_time_step=11
time_val=np.linspace(0, num_time_step, num_time_step)
sld_in=1
sld_out=0

#
mode='shrink'

shrnk_rate=-1

    

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
dyn_file='multi_grain_'+ mode
dyn_folder=os.path.join(res_folder,dyn_file)
os.system('rm -r {0}*'.format(dyn_folder))
os.makedirs(dyn_folder, exist_ok=True)

simu_save_t1=time.perf_counter()
if mode=='shrink':
    rad_med=35
    sigma=0.4
    num_bin=10
    print('creating simulation with schrinking grain')
    print('data will be saved in folder {0}'.format(dyn_folder))
    print('median radius : {0}, \u03C3 : {1}'.format(rad_med, sigma))  
    mu=np.log(rad_med)
    p=sigma*rad_med
    start=15#69-2*p
    end=rad_med+2*p

    rad_val = np.linspace(start,end,num_bin)
    rad_val=np.round(rad_val, 3)
    rad_num = (1/rad_val*sigma)*np.exp(-0.5*((np.log(rad_val)-mu)/(sigma))**2)
    rad_num=np.round(100*(rad_num/np.sum(rad_num)))
    
    print('\n############ initial status ##############')
    for i in range(len(rad_val)):
        print('{0} grains of raidus {1}'.format(rad_num[i], rad_val[i]))
    print('##########################################\n')
    
    
    rdistfile_name=os.path.join(res_folder, 'rdist_{0}_{1}_{2}.h5'
                                .format(rad_med, str(sigma).replace('.','p'), num_bin))
    if os.path.exists(rdistfile_name):
        print('{0} already exists'.format(rdistfile_name))
        os.system('cp -r {0} {1}'.format(rdistfile_name, dyn_folder))
        rdistfile=h5py.File(rdistfile_name, 'r')
        radii=rdistfile['radii'][:]
        r_dist=rdistfile['r_dist'][:]
        rdistfile.close()
        
    else:
        r_dist_t1=time.perf_counter()
        r_dist, radii = sim.sph_multigrain_loc_3d(nodes,rad_val,rad_num,sld_in,sld_out, length_a)
        rdistfile=h5py.File(rdistfile_name, 'w')
        rdistfile['r_dist']=r_dist
        rdistfile['radii']=radii
        rdistfile['rad_val']=rad_val
        rdistfile['rad_num']=rad_num
        rdistfile['rad_med']=rad_med
        rdistfile['sigma']=sigma
        rdistfile['num_bin']=num_bin
        rdistfile.close()
        os.system('cp -r {0} {1}'.format(rdistfile_name, dyn_folder))
        r_dist_t2=time.perf_counter()
        print('Calculation of distribution of radius took {0} sec'.format(r_dist_t2-r_dist_t1))
        
neutron_count=np.zeros(num_time_step)


for t in range(len(time_val)):
    time_dir=os.path.join(dyn_folder,'t{0:0>3}'.format(t))
    if mode=='shrink':
        #continue
        radii_t=radii+shrnk_rate*t
        print('radius values at time {0}:'.format(round(time_val[t])))
        idx=0
        rad_num_cum=np.cumsum(rad_num)
        for j in range(len(radii)):
            if (j%rad_num_cum[idx])==0:
                idx+=1    
                print('\t radius of {0} became {1}:'.format(round(radii[j]), radii_t[j]))        
        simu_t1=time.perf_counter()
        #### simulation start ###
        sld_dyn = sim.sph_multigrain_3d(nodes,radii_t, r_dist, sld_in,sld_out)
        #sld_dyn = sim.sph_grain_3d(nodes,[length_a/2,length_b/2,length_c/2],rad_t,sld_in,sld_out)
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
    data_file['radius']=radii_t
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




