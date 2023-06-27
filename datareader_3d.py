#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 12:24:56 2023

@author: amajumda
"""
from lib import struct_gen as sg
from lib import plotter as pltr
from lib import simulation as sim
from lib import processing as procs
from lib import datasaver as dsv

import os
import numpy as np


"""
configuration of the structure 
"""

#box side lengths
length_a=20
length_b=20
length_c=20
# number of cells in each direction
nx=20
ny=20
nz=20
# ball radius
radius=5
# sld of ball and box (sld_in = ball, sld_out = box) 
sld_in=1
sld_out=0

#folder to be read
folder='./data/'+str(length_a)+'_'+str(length_b)+'_'+str(length_c)+\
    '_'+str(nx)+'_'+str(ny)+'_'+str(nz)+\
        '_'+str(radius)+'_'+str(sld_in)+'in_'+str(sld_out)+'out/'
data_file='data.h5'
data_file_full=os.path.join(folder,data_file)
sass_folder=folder+'sassena/'
os.makedirs(sass_folder, exist_ok=True)

# retrieve all data from data.h5
node, cell, sld_cell, sld_node, con = procs.hdfreader(data_file_full)

#categorize sld
cat_sld_cell, cat_sld_cell_idx = procs.categorize_prop(sld_cell,[0],10)

pltr.colorplot_cell_3d(cell,cat_sld_cell,nx,ny,nz)

# generate pdb and dcd
dsv.pdb_dcd_gen(cell, cat_sld_cell, np.min(cat_sld_cell), np.max(cat_sld_cell), 10,sass_folder+'pdb_dcd/')
#generate scatter.xml
dsv.scatterxml_generator(sass_folder, sigfile='signal.h5')
#generate database
dsv.database_generator(np.min(cat_sld_cell), np.max(cat_sld_cell), ndiv=10, database_dir=sass_folder+'database/')
# slurm script generator
dsv.slurm_script_gen(sass_folder,24,1,xml_file='scatter.xml',sas=' /data/data/amajumda/sass_paper/sassena_Glab/sassena/compile_boxcut_img/sassena')
