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

import logging

# Create a logger
logger = logging.getLogger('my_logger')

# Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR)
logger.setLevel(logging.INFO)

# Create a file handler to write log messages to a file
file_handler = logging.FileHandler('datareader.log')

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
logging.info('reading {0} ...'.format(data_file_full))
print('reading {0} ...'.format(data_file_full))
node, cell, sld_cell, sld_node, con = procs.hdfreader(data_file_full)
logging.info('read nodes, cells, sld of node and cells from {0} ...'.format(data_file_full))
print('read nodes, cells, sld of node and cells from {0} ...'.format(data_file_full))

#categorize sld
logging.info('categorizing slds')
print('categorizing slds')
cat_sld_cell, cat_sld_cell_idx = procs.categorize_prop(sld_cell,[0],10)
logging.info('completed categorizing')
print('completed categorizing')

pltr.colorplot_cell_3d(cell,cat_sld_cell,nx,ny,nz)

# generate pdb and dcd
logging.info('generating pdb, dcd, scatter.xml, database, slurm script (specific to mlz cluster) in {0}'.format(sass_folder))
print('generating pdb, dcd, scatter.xml, database, slurm script (specific to mlz cluster) in {0}'.format(sass_folder))
dsv.pdb_dcd_gen(cell, cat_sld_cell, np.min(cat_sld_cell), np.max(cat_sld_cell), 10, sass_folder+'pdb_dcd/')
logging.info('created pdb and dcd in {0}'.format(sass_folder+'pdb_dcd/'))
print('created pdb and dcd in {0}'.format(sass_folder+'pdb_dcd/'))


#generate scatter.xml
dsv.scatterxml_generator(sass_folder, sigfile='signal.h5')
logging.info('created scatter.xml in {0}'.format(sass_folder))
print('created scatter.xml in {0}'.format(sass_folder))

#generate database
dsv.database_generator(np.min(cat_sld_cell), np.max(cat_sld_cell), ndiv=10, database_dir=sass_folder+'database/')
logging.info('created database in {0}'.format(sass_folder))
print('created database in {0}'.format(sass_folder))

# slurm script generator
dsv.slurm_script_gen(sass_folder,24,1,xml_file='scatter.xml',sas='/data/data/amajumda/sass_paper/sassena_Glab/sassena/compile_boxcut_img/sassena')
logging.info('created slurm script in {0}'.format(sass_folder))
print('created slurm script in {0}'.format(sass_folder))