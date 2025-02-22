"""
This creates a 3D strcuture and saves it in the data folder

Created on Fri Jun 23 10:28:09 2023

@author: amajumda
"""
import sys
import os
lib_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(lib_dir)

from lib import struct_gen as sg
from lib import plotter as pltr
from lib import simulation as sim
from lib import processing as procs
from lib import datasaver as dsv
from lib import scatt_cal as scatt



import os
import time
import argparse
import sys
import xml.etree.ElementTree as ET
import numpy as np
import subprocess
import matplotlib.pyplot as plt
import h5py
import imageio.v2 as imageio
import mdtraj as md
from matplotlib.patches import Rectangle

# function
def J1(x):
    if x==0:
        return 0
    else:
        return (np.sin(x)-x*np.cos(x))/x**2

def ball (qmax,qmin,Npts,scale,bg,sld,sld_sol,rad):
    vol=(4/3)*np.pi*rad**3
    # SLD unit 10^-5 \AA^-2
    del_rho=sld-sld_sol
    q_arr=np.linspace(qmin,qmax,Npts)
    FormFactor=np.zeros(len(q_arr))
    for i in range(len(q_arr)):
        q=q_arr[i]
        if q==0:
            # Form factor unit 10^-5 \AA
            FormFactor[i]=1
        else:
            # Form factor unit 10^-5 \AA
            FormFactor[i]=3*vol*del_rho*J1(q*rad)/(q*rad)
    # Intensity unit 10^-10 \AA^2
    Iq_arr = ((scale)*np.abs(FormFactor)**2+bg)
    return Iq_arr, q_arr


#timer counter initial
tic = time.perf_counter()

"""
read input from xml file
"""

### struct xml ###

xml_folder='../xml/'

"""
Input data
"""
### struct.xml entries ###

# box side lengths (float values)
length_a= 40.  
length_b=40. 
length_c=40. 
# number of cells in each direction (int values)
nx= 40
ny= 40
nz= 40
# calculate mid point of structure (simulation box)
mid_point=np.array([length_a/2, length_b/2, length_c/2])

### sim xml entries ###

# model name
sim_model= 'fs' #root.find('model').text

# simulation parameters
## time
dt= 1. 
t_end= 10.
t_arr=np.arange(0,t_end+dt, dt)
## ensemble
n_ensem=1 #int(root.find('sim_param').find('n_ensem').text)

### model xml entries ###

# model params
ball_rad=10
sig_0=0
sig_end=4
sld_in=2
sld_out=1
# dir name for model param
model_param_dir_name = ('rad' + '_' + str(ball_rad) + '_' + 
                        'sig_0' + '_' + str(sig_0) + '_' +
                        'sig_end' + '_' + str(sig_end) + '_' +
                        'sld_in' + '_' + str(sld_in) + '_' +
                        'sld_out' + '_' + str(sld_out)).replace('.', 'p')

### scatt_cal xml entries ###

# decreitization params
# number of categories and method of categorization
num_cat= 101
method_cat='extend'

# scatt_cal params
signal_file='signal.h5'
resolution_num=10
start_length=0.
end_length=1.
num_points=100
scan_vec_x=1.
scan_vec_y=0.
scan_vec_z=0.
scan_vector=[scan_vec_x, scan_vec_y, scan_vec_z]
scatt_settings='cat_' + method_cat + '_' + str(num_cat) + 'Q_' \
    + str(start_length) + '_' + str(end_length) + '_' + 'orien_' + '_' + str(resolution_num)
scatt_settings=scatt_settings.replace('.', 'p')


"""
read folder structure
"""

# folder structure
## mother folder name
### save length values as strings
### decimal points are replaced with p
length_a_str=str(length_a).replace('.','p')
length_b_str=str(length_a).replace('.','p')
length_c_str=str(length_a).replace('.','p')

### save num_cell values as strings
nx_str=str(nx)
ny_str=str(ny)
nz_str=str(nz)
# element type
el_type='lagrangian'
el_order=1
el_order_str=str(el_order)


mother_dir_name = length_a_str+'_' + length_b_str+'_' + length_c_str\
      + '_' + nx_str + '_' + ny_str+'_' + nz_str+'_'+el_type+'_'+ el_order_str
mother_dir = os.path.join('../data/', mother_dir_name)

# read structure info
data_file=os.path.join(mother_dir, 'structure/struct.h5')

sim_dir=os.path.join(mother_dir, 'simulation')

# folder name for model
model_dir_name= (sim_model + '_tend_' + str(t_end) + '_dt_' + str(dt) \
    + '_ensem_' + str(n_ensem)).replace('.','p')
model_dir=os.path.join(sim_dir,model_dir_name)

# folder name for model with particular run param
model_param_dir=os.path.join(model_dir,model_param_dir_name)

# create folder for figure (one level up from data folder)
figure_dir=os.path.join(mother_dir, '../../figure/')
os.makedirs(figure_dir, exist_ok=True)
## folder for this suit of figures
plot_dir=os.path.join(figure_dir, sim_model)
os.makedirs(plot_dir, exist_ok=True)

if os.path.exists(model_param_dir):
    print('model folder exists')
else:
    print('model folder does not exist')

for i in range(len(t_arr)):
    t=t_arr[i]
    print(t)
    # time_dir name
    t_dir_name='t{0:0>3}'.format(i)
    t_dir=os.path.join(model_param_dir, t_dir_name)
    for j in range(n_ensem):
        idx_ensem=j
        # create ensemble dir
        ensem_dir_name='ensem{0:0>3}'.format(idx_ensem)
        ensem_dir=os.path.join(t_dir, ensem_dir_name)

        # create scatt dir
        scatt_dir_name='scatt_cal_'+ scatt_settings
        scatt_dir=os.path.join(ensem_dir, scatt_dir_name)


        """
        read pseudo atom info from scatt_cal.h5
        """
        # read node sld
        # save pseudo atom info
        scatt_cal_dir_name='scatt_cal_' + scatt_settings
        scatt_cal_dir=os.path.join(ensem_dir, scatt_cal_dir_name)
        scatt_cal_data_file_name='scatt_cal.h5'
        scatt_cal_data_file=os.path.join(scatt_cal_dir, scatt_cal_data_file_name)
        scatt_cal_data=h5py.File(scatt_cal_data_file,'r')
        node_pos=scatt_cal_data['node_pos'][:]
        node_sld=scatt_cal_data['node_sld'][:]
        pseudo_pos=scatt_cal_data['pseudo_pos'][:]
        pseudo_b=scatt_cal_data['pseudo_b'][:]
        pseudo_b_cat_val=scatt_cal_data['pseudo_b_cat_val'][:]
        pseudo_b_cat_idx=scatt_cal_data['pseudo_b_cat_idx'][:]
        scatt_cal_data.close()

        # determine sld min and max for plotting
        sld_min=np.min(node_sld,0)
        sld_max=np.max(node_sld,0)


        if idx_ensem==0:
            print('plotting for the first ensemble')
            # plotting node SLD
            ## cutting at z = cut_frac * length_z
            if el_type=='lagrangian':
                num_node_x=el_order*nx+1
                num_node_y=el_order*ny+1
                num_node_z=el_order*nz+1
            cut_frac=0.5
            node_pos_3d=node_pos.reshape(num_node_x, num_node_y, num_node_z, 3)
            z_idx= np.floor(cut_frac*(num_node_z)).astype(int)
            y_idx= np.floor(cut_frac*(num_node_z)).astype(int)
            y_val=node_pos_3d[0, y_idx, z_idx , 1]
            z_val=node_pos_3d[0, y_idx, z_idx , 2]
            # plot nodes
            # image plot
            ## .T is required to exchange x and y axis 
            ## origin is 'lower' to put it in lower left corner 
            node_sld_3d=node_sld.reshape(num_node_x, num_node_y, num_node_z)
            node_pos_1d=node_pos_3d[:,y_idx,z_idx,0]
            node_sld_1d=node_sld_3d[:,y_idx,z_idx]
            # img = ax.plot(node_pos_1d, node_sld_1d)
            x_line_1=30*np.ones(20)
            y_line_1=np.linspace(sld_in, sld_out,20)
            x_line_2=np.linspace(0, 40,20)
            y_line_2=0.5*(sld_in+sld_out)*np.ones(20)
            print(len(node_pos_1d))
            print(len(node_sld_1d))

            plt.plot(node_pos_1d, node_sld_1d,'o')
            plt.plot(x_line_1, y_line_1,'-')
            plt.plot(x_line_2, y_line_2,'-')
            plt.show()

            #plot pseudo atoms
            cut_frac=0.5
            pseudo_pos_3d=pseudo_pos.reshape(nx, ny, nz, 3)
            pseudo_b_3d=pseudo_b.reshape(nx, ny, nz)
            z_idx_pseudo= nz//2-1
            y_idx_pseudo= ny//2-1
            z_val_pseudo=pseudo_pos_3d[0, y_idx_pseudo, z_idx_pseudo , 2]
            y_val_pseudo=pseudo_pos_3d[0, y_idx_pseudo, z_idx_pseudo , 1]
            pseudo_pos_x=pseudo_pos_3d[:,y_idx_pseudo,z_idx_pseudo,0]
            pseudo_b_x=pseudo_b_3d[:,y_idx_pseudo,z_idx_pseudo]
            plt.plot(pseudo_pos_x, pseudo_b_x,'o')
            plt.plot(x_line_1, y_line_1,'-')
            plt.plot(x_line_2, y_line_2,'-')
            plt.show()

            # plot categorized pseudo atoms
            pseudo_b_cat_val_3d=pseudo_b_cat_val.reshape(nx, ny, nz)
            pseudo_b_cat_val_x=pseudo_b_cat_val_3d[:,y_idx_pseudo,z_idx_pseudo]
            plt.plot(pseudo_pos_x, pseudo_b_cat_val_x,'o')
            plt.plot(x_line_1, y_line_1,'-')
            plt.plot(x_line_2, y_line_2,'-')
            plt.show()