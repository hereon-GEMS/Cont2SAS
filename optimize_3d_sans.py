#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 19:12:46 2023

@author: amajumda
"""
from lib import struct_gen as sg
from lib import plotter as pltr
from lib import simulation as sim
from lib import processing as procs
from lib import datasaver as dsv

import logging
logging.basicConfig(filename='log_optimize.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s : %(message)s')
import os
import shutil
import numpy as np
import sys

#create intial structure
length_a=20
length_b=20
length_c=20
nx=20
ny=20
nz=20
radius=5
sld_in=1
sld_out=0
ndiv=10 # 10 categories


datafilename='data.h5'
sigfilename='sassena/signal.h5'
folder='data/{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}in_{8}out/'\
    .format(length_a,length_b,length_c,nx,ny,nz,radius,sld_in,sld_out)
    
if not os.path.exists(folder):
    print('reference data does not exist, please run ball in box to create reference data with appropiate inputs')
    sys.exit()
    
datafilepath=os.path.join(folder, datafilename)
sigfilepath=os.path.join(folder, sigfilename)

# read data
q, fq0=procs.signalreader(sigfilepath)
node_ref, cell_ref, sld_cell_ref, sld_node_ref, con_ref = procs.hdfreader(datafilepath)

#create sld with respect to r
def runner(r):
    sld_it=sim.sph_grain_3d(node_ref,[length_a/2,length_b/2,length_c/2],r,1,0)
    pltr.colorplot_node_3d(node_ref,sld_it,nx,ny,nz,save_plot=False,save_dir='')
    cell_it, cell_sld_it = procs.node2cell_3d (node_ref,con_ref, [sld_it], [0], nx, ny, nz)
    pltr.colorplot_cell_3d(cell_it,cell_sld_it,nx,ny,nz,save_plot=False,save_dir='')
    cat_sld_cell_it, cat_idx_it=procs.categorize_prop(cell_sld_it,[0],ndiv)
    iteration_dir=os.path.join(folder, 'iterations')
    os.makedirs(iteration_dir,exist_ok=True)
    i=0
    while os.path.exists(os.path.join(iteration_dir,'step_{0:0>3}'.format(i))):
        i+=1
        print(i)
    stepdir=os.path.join(iteration_dir,'step_{0:0>3}'.format(i))
    os.makedirs(stepdir,exist_ok=True)
    os.system('cp -r {0}/sassena/scatter.xml {1}/'.format(folder,stepdir))
    os.system('cp -r {0}/sassena/database {1}/'.format(folder,stepdir))
    dsv.pdb_dcd_gen(cell_it, cat_sld_cell_it, 0, 1, ndiv, os.path.join(stepdir,'pdb_dcd'))
    root_dir=os.getcwd()
    print(root_dir)
    os .chdir(stepdir)
    print('working directory changed to {0}'.format(os.getcwd()))
    slurm=False
    if os.path.exists('signal.h5'):
        i=0
        if os.path.exists('signal_{0:0>6}.h5'.format(i)):
            i=i+1
        os.system('mv signal.h5 signal_{0:0>6}.h5'.format(i))
        print('removed old result signal file and saved it to signal_{0:0>6}.h5'.format(i))
    if slurm:
        os.system('sbatch run_slurm_icc')
        print('running sassena')
    else:
        os.system('mpirun -np 4 sassena')
    os.chdir(root_dir)
    print(os.path.join(stepdir,'signal.h5'))
    q,fq0_it=procs.signalreader(os.path.join(stepdir,'signal.h5'))
    return fq0_it-fq0


from scipy.optimize import minimize
res = minimize(runner, 4)
print(res.x)

"""
node, cell, sld_cell, sld_node, con = procs.hdfreader(datafilepath)
cat_sld_cell, cat_sld_cell_idx = procs.categorize_prop(sld_cell,[0],10)
pltr.colorplot_cell_3d(cell,cat_sld_cell,nx,ny,nz,show_plot=False)
dsv.pdb_dcd_gen(cell, cat_sld_cell, np.min(cat_sld_cell), np.max(cat_sld_cell), 10, folder+'pdb_dcd/')
q, fq0 = procs.signalreader(sigfilepath)
"""