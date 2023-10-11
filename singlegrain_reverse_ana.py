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
ny=10
nz=10


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
    r_slope=-1
    slope = fit.sph_shrnk_slope(t, count, q, 35)
    n_count = fit.sph_shrnk_dyn(t, count, q, 35, slope)
    print(r_slope)
    

plt.plot(t, count)
plt.plot(t, n_count)
#plt.loglog(q, fq0)
#plt.loglog(q[1:], fit.sph_fq(1,q,35)[1:])