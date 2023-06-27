#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 12:24:56 2023

@author: amajumda
"""
from src.struct_gen import *
from src.plotter import *
from src.simulation import *
from src.processing import *
from src.datasaver import *

import os
import numpy as np


length_a=20
length_b=20
length_c=20
nx=20
ny=20
nz=20
radius=5
sld_in=1
sld_out=0
folder='./data/'+str(length_a)+'_'+str(length_b)+'_'+str(length_c)+\
    '_'+str(nx)+'_'+str(ny)+'_'+str(nz)+\
        '_'+str(radius)+'_'+str(sld_in)+'in_'+str(sld_out)+'out/'
data_file='data.h5'
data_file_full=os.path.join(folder,data_file)
node, cell, sld_cell, sld_node, con = hdfreader(data_file_full)
cat_sld_cell, cat_sld_cell_idx = categorize_prop(sld_cell,[0],10)
colorplot_cell_3d(cell,cat_sld_cell,nx,ny,nz,show_plot=False)
pdb_dcd_gen(cell, cat_sld_cell, np.min(cat_sld_cell), np.max(cat_sld_cell), 10,folder+'pdb_dcd/')
scatterxml_generator(folder, sigfile='signal.h5')
database_generator(np.min(cat_sld_cell), np.max(cat_sld_cell), ndiv=10, database_dir=folder+'database/')
