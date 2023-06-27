#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 19:12:46 2023

@author: amajumda
"""
from src import struct_gen as sg
from src import plotter as pltr
from src import simulation as sim
from src import processing as procs
from src import datasaver as dsv

import logging
logging.basicConfig(filename='log_optimize.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s : %(message)s')
import os
import shutil
import numpy as np

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
sigfilename='signal.h5'
folder='data/{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}in_{8}out'\
    .format(length_a,length_b,length_c,nx,ny,nz,radius,sld_in,sld_out)
datafilepath=os.path.join(folder, datafilename)#'data/{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}in_{8}out/{9}'\
    #.format(length_a,length_b,length_c,nx,ny,nz,radius,sld_in,sld_out,datafilename)
sigfilepath=os.path.join(folder, sigfilename)#'data/{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}in_{8}out/{9}'\
    #.format(length_a,length_b,length_c,nx,ny,nz,radius,sld_in,sld_out,sigfilename)
node, cell, sld_cell, sld_node, con = procs.hdfreader(datafilepath)
cat_sld_cell, cat_sld_cell_idx = procs.categorize_prop(sld_cell,[0],10)
pltr.colorplot_cell_3d(cell,cat_sld_cell,nx,ny,nz,show_plot=False)
dsv.pdb_dcd_gen(cell, cat_sld_cell, np.min(cat_sld_cell), np.max(cat_sld_cell), 10, folder+'pdb_dcd/')
q, fq0 = procs.signalreader(sigfilepath)