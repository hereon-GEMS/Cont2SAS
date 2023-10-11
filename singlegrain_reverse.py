#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 16:07:05 2023

@author: amajumda
"""
from lib import struct_gen as sg
from lib import plotter as pltr
from lib import simulation as sim
from lib import processing as procs
from lib import datasaver as dsv
from lib import fitter as fit

import os
import time
import argparse
import sys
import logging
import h5py
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Create a logger
logger = logging.getLogger('single_rev_logger')

# Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR)
logger.setLevel(logging.INFO)

# Create a file handler to write log messages to a file
file_handler = logging.FileHandler('single_rev.log')

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
nx=10
ny=nx
nz=nx
# cell volume
cell_vol=(length_a/nx)*(length_a/ny)*(length_a/nz)

# time steps
num_time_step=11
sld_in=1
sld_out=0


#
mode='shrink'
    

# result folder structure
os.makedirs('data', exist_ok=True)
res_folder=os.path.join('./data/',
                        str(length_a)+'_'+str(length_b)+'_'+str(length_c)+'_'+
                        str(nx)+'_'+str(ny)+'_'+str(ny))


# read data from signal file 
sig_file=os.path.join(res_folder, 'single_atom_'+mode+'/t{0:0>3}/signal.h5'.format(0)) 
q, fq0 = procs.signalreader(sig_file)
sig_file=os.path.join(res_folder, 'single_atom_'+mode+'/count.h5') 
t, count = procs.countreader(sig_file)

# t=0 fitting
if mode == 'shrink':
    r_shrnk = fit.sph_shrnk_r(q, fq0)
    print(r_shrnk)
    
#node cell con retrieval
# get nodes, cells, connectivity
data_filename=os.path.join(res_folder,'data.h5')
data_file=h5py.File(data_filename, 'r')
nodes=data_file['nodes'][:]
cells=data_file['cells'][:]
con=data_file['connectivity'][:]
data_file.close()


os.system('rm -r {0}'.format(os.path.join(res_folder, 'single_atom_'+mode+'/iteration_*')))

def func(time_step, slope):
    print('slope : {}'.format(slope))
    num_time_step=len(time_step)
    
    # creation of iteration folder
    j=0
    while os.path.exists(os.path.join(res_folder, 'single_atom_'+mode+'/iteration_{0:0>3}'.format(j))):
        j+=1
    
    print('%%%%%%%%% iteration : {0} %%%%%%%%%'.format(j))
    it_dir=os.path.join(res_folder, 'single_atom_'+mode+'/iteration_{0:0>3}'.format(j))
    os.makedirs(it_dir, exist_ok=True)
    
    neutron_count=np.zeros(num_time_step)
    for t in range(num_time_step):
        print ('time : {}'.format(t ))
        it_t_dir=os.path.join(it_dir, 't{0:0>3}'.format(t))
        os.makedirs(it_t_dir, exist_ok=True)
        if mode == 'shrink':
            rad_t=r_shrnk+slope*t
            print('\t simulation starts')  
            #### simulation start ###
            sld_dyn = sim.sph_grain_3d(nodes,[length_a/2,length_b/2,length_c/2],rad_t,sld_in,sld_out)
            sld_dyn_cell=procs.node2cell_3d_t(nodes , cells, con, sld_dyn, nx, ny, nz, cell_vol)
            sld_dyn_cell_cat, cat = procs.categorize_prop_3d_t(sld_dyn_cell, 10)
            #### simulation end ###
            print('\t simulation end')
            
        ### data saving start ###
        print('\t saving simulation data')
        data_file_full=os.path.join(it_t_dir,'data_{0:0>3}.h5'.format(t))
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
        print('\t saving simulation data completed')
        ### data saving end ###
        
        ### pdb dcd generation ###
        print('\t preparing sassena run')
        pdb_dcd_dir=os.path.join(it_t_dir,'pdb_dcd')
        os.makedirs(pdb_dcd_dir, exist_ok=True)
        dsv.pdb_dcd_gen_opt1(cells, sld_dyn_cell_cat, cat, pdb_dcd_dir)
        #print('saving pdb and dcd file completed')
    
        ### scatter.xml generate ###
        dsv.scatterxml_generator(it_t_dir, sigfile='signal.h5')
        
        ### database generator ###
        db_dir=os.path.join(it_t_dir,'database')
        dsv.database_generator(min(sld_dyn_cell_cat), max(sld_dyn_cell_cat), ndiv=10, database_dir=db_dir)
        print('\t preparing sassena run completed')
        
        ### sassena runner ###
        #os.system('cd ' + time_dir)
        print('\t running sassena')
        parent_dir=os.getcwd()
        os.chdir(os.path.join(parent_dir,it_t_dir))
        #print(os.getcwd())
        os.system('mpirun -np 8 sassena > sass_output.log 2>&1')
        os.chdir(parent_dir)
        print('\t sassena run completed')
        #print(os.getcwd())
        
        ### 
        print('\t calculating neutron count')
        sigfile_name=os.path.join(it_t_dir,'signal.h5')
        sigfile=h5py.File(sigfile_name, 'r')
        q_vec_t=np.sqrt(np.sum(sigfile['qvectors'][:]**2,axis=1))
        q_args=np.argsort(q_vec_t)
        fq0_t=np.sqrt(np.sum(sigfile['fq0'][:]**2,axis=1))
        q_vec_t=q_vec_t[q_args]
        fq0_t=fq0_t[q_args]
        sigfile.close()
        neutron_count_t=0
        for i in range(len(q_vec_t)-1):
            del_q=q_vec_t[i+1]-q_vec_t[i]
            neutron_count_t+=0.5*del_q*(fq0_t[i+1]+fq0_t[i])
        #print('neutron count is '+ str(neutron_count_t))
        neutron_count[t]=neutron_count_t
        print('\t calculating neutron count completed')
    # save count vs t file
    countdatafile_name=os.path.join(it_dir, 'count.h5')
    countdatafile=h5py.File(countdatafile_name, 'w')
    countdatafile['count']=neutron_count
    countdatafile['time']=np.linspace(0,num_time_step-1,num_time_step)
    countdatafile['shrnk_rate']=slope[0]
    countdatafile.close()
    print('\t this time step completed')
    return neutron_count

timestep=np.linspace(0,10,11)
def chi_sq(slope,exp=count, timestep=timestep):
    #timestep=np.linspace(0,10,11)
    count_it=func(timestep, slope)
    chi_sq=np.sum(((count_it-exp)/exp)**2)
    print(chi_sq)
    #print('current radius is ' + r)
    return chi_sq

def opt_func(slope):
    return chi_sq(slope)

from scipy.optimize import minimize
timestep=np.linspace(0,10,11)
res = minimize(opt_func, -2.0, method='Nelder-Mead', tol=1, options={'maxiter': 50})

# timestep=np.linspace(0,10,11)
# popt, pcov = curve_fit(func, timestep, count)

#print('slope : {0}'.format(*popt))
plt.plot(t, count)
plt.plot(timestep, func(timestep, res.x))
# plt.plot(t, n_count)
#plt.plot(q, fq0)
#plt.plot(q[1:], (fq0-fit.sph_fq(1,q,35))[1:])