#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 15:24:14 2023

@author: amajumda
"""
# from lib import struct_gen as sg
# from lib import plotter as pltr
from lib import simulation as sim
from lib import processing as procs
from lib import datasaver as dsv
from lib import plotter as pltr

import os
import time
# import argparse
# import sys
# import logging
import h5py
# import subprocess
import numpy as np
import math
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import imageio


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
t_end=10
dt=1
num_time_step=math.floor(t_end/dt+1)
time_arr= np.arange(0,t_end+dt,dt)

# print(num_time_step)

#
mode='fs'
   

# result folder structure
os.makedirs('data', exist_ok=True)
res_folder=os.path.join('./data/',
                        str(length_a)+'_'+str(length_b)+'_'+str(length_c)+'_'+
                        str(nx)+'_'+str(ny)+'_'+str(nz))
dyn_file='single_grain_'+ mode + '_' + str(t_end) + '_' + str(dt).replace('.','p')
dyn_folder=os.path.join(res_folder,dyn_file)

for i in range(num_time_step):
    #print(i)
    data_file_t=os.path.join(dyn_folder,'t{0:0>3}/data_{0:0>3}.h5'.format(i))
    # file=h5py.File(data_file_t,'r')
    # #mode=file['mode'][()]
    # print(type(file['radius'][()]))
    # file.close()
    
    if mode=='shrink' or mode=='gg':
        nodes, nodeprop, cell, cellprop, catcellprop, catcell,\
            mode, grain_sld, env_sld, rad_t = dsv.sim_read(data_file_t)
    if mode=='diffuse':
        nodes, nodeprop, cell, cellprop, catcellprop, catcell,\
            mode, grain_sld, env_sld, rad, D = dsv.sim_read(data_file_t)
    if mode=='fs':
        nodes, nodeprop, cell, cellprop, catcellprop, catcell,\
            mode, grain_sld, env_sld, rad, sig_t = dsv.sim_read(data_file_t)

    pltr.p_scatter_plot_mesh(cell, cellprop, nodes, i, mode, dyn_folder, grain_sld, env_sld)
    
    pltr.img_plot_mesh(cell, nodes, nodeprop, i, dyn_folder, mode, grain_sld, env_sld)

images=[]
images1=[]
for i in range(num_time_step):
    plot_folder_t=os.path.join(dyn_folder,'t{0:0>3}/images'.format(i))
    os.makedirs(plot_folder_t, exist_ok=True)
    plot_file=os.path.join(plot_folder_t,'node_{1:0>3}_{0}.png'.format(mode,i))
    #print(plot_file)
    images.append(imageio.imread(plot_file))
    plot_file=os.path.join(plot_folder_t,'cat_cell_{1:0>3}_{0}.png'.format(mode,i))
    #print(plot_file)
    images1.append(imageio.imread(plot_file))
imageio.mimsave(os.path.join(dyn_folder,'{0}_node.gif'.format(mode)), images, fps=2, loop=0)
imageio.mimsave(os.path.join(dyn_folder,'{0}_pscatter.gif'.format(mode)), images1, fps=2, loop=0)
    # nodes, nodeprop, cell, cellprop, \
    #     catcellprop, catcell, mode, grain_sld, env_sld, rad, D=dsv.sim_read(data_file_t)
    #print(file)

# # get nodes, cells, connectivity
# data_filename=os.path.join(res_folder,'data.h5')
# #nodes, cells, con = dsv.mesh_read(data_filename)

# # create dolder for dynamics
# dyn_file='single_grain_'+ mode + '_' + str(t_end) + '_' + str(dt).replace('.','p')
# dyn_folder=os.path.join(res_folder,dyn_file)
# #os.makedirs(dyn_folder, exist_ok=True)
# #os.system('rm -r {0}/*'.format(dyn_folder))
# #simu_save_t1=time.perf_counter() # time counter for simulation

# # if mode=='shrink':
# #     print('creating simulation with schrinking grain')
# # if mode== 'diffuse':
# #     print('creating simulation with diffusion into grain')
# # if mode== 'gg':
# #     print('creating simulation with growing grain')

# # print('data will be saved in folder {0}'.format(dyn_folder))    
 

# for i in range(10):
        
#         file=os.path.join(folder,'{0}/t{1:0>3}/data_{1:0>3}.h5'.format(mode,i))
#         # read file
#         file_read = h5py.File(file, 'r')
#         cell = file_read['cell'][:]
#         node = file_read['node'][:]
#         node_prop = file_read['nodeprop'][:]
#         cell_prop = file_read['catcellprop'][:]
        
#         #read cells
#         x = cell[:,0]
#         y = cell[:,1]
#         z = cell[:,2]
#         sld = (7+1.565)*cell_prop[:]-1.565
#         #print(np.max(sld))
#         # 
#         x_red = x[z==10.25]
#         y_red = y[z==10.25]
#         z_red = z[z==10.25]
#         sld_red=sld[z==10.25]
        
#         # plot cell values
#         nx=40
#         ny=nx
#         nz=nx
#         divx=20/nx
#         divy=20/ny
#         divz=20/nz
#         area=np.ones(nx*ny)
#         fig, ax = plt.subplots(figsize=(10,8)) 

#         scatt_plot = ax.scatter(x_red, y_red, s=40, c=sld_red, edgecolors=None, cmap='viridis', vmin=-1.565, vmax=7)

#         ax.set_title('t = {0} s, z = 10.25 $\AA$'.format(i), fontsize=20)
#         ax.set_xlabel('x [$\AA$]', fontsize=20)
#         ax.set_ylabel('y [$\AA$]', fontsize=20)
#         #cbar=fig.colorbar(images)
#         #cbar.ax.tick_params(labelsize=20)
#         #ax.set_aspect('equal', adjustable='box')
#         ax.tick_params(which='both', width=1, length=10, labelsize=20)
#         ax.set_xlim([0,20])
#         ax.set_ylim([0,20])
#         cbar=fig.colorbar(scatt_plot)
#         cbar.ax.tick_params(labelsize=20)
#         #ax.set_title('t = {0}'.format(i))
#         ax.set_aspect('equal', adjustable='box')

#         for ix in range(nx):
#             for jy in range(ny):
#                 #print(i*0.5)
#                 rect = patches.Rectangle((jy*divy, ix*divx), divy, divx, linewidth=0.5, edgecolor='k', facecolor='none')
#                 ax.add_patch(rect)
#         plot_file=os.path.join(folder,'{0}/t{1:0>3}/cat_cell_{1:0>3}_{0}.pdf'.format(mode,i))
#         plt.savefig(plot_file, bbox_inches='tight')
#         plt.close()
        
#         # plot imshow
#         sld = node_prop[:]#[0,:]
#         sld_shape=sld.reshape(41,41,41)*(7+1.565)-1.565
#         print(np.max(sld_shape))
#         fig, ax = plt.subplots(figsize=(10,8))
#         extent=[0, 20, 0, 20]
#         images = ax.imshow(sld_shape[:,:,21], cmap='viridis', vmin=-1.565, vmax=7, extent=extent, interpolation='bilinear')#, vmin=0, vmax=20)
#         #print(np.max(sld_shape[:,:,21]))
        
#         ax.set_title('t = {0} s, z = 10 $\AA$'.format(i), fontsize=20)
#         ax.set_xlabel('x [$\AA$]', fontsize=20)
#         ax.set_ylabel('y [$\AA$]', fontsize=20)
#         cbar=fig.colorbar(images)
#         cbar.ax.tick_params(labelsize=20)
#         ax.set_aspect('equal', adjustable='box')
#         ax.tick_params(which='both', width=1, length=10, labelsize=20)
#         #plt.xticks(fontsize=30)
#         #plt.yticks(fontsize=30)
        
#         for axis in ['top','bottom','left','right']:
#             ax.spines[axis].set_linewidth(0.5)
        

#         for ix in range(nx):
#             for jy in range(ny):
#                 #print(i*0.5)
#                 rect = patches.Rectangle((jy*divy, ix*divx), divy, divx, linewidth=0.5, edgecolor='k', facecolor='none')
#                 ax.add_patch(rect)
        
#         plot_file=os.path.join(folder,'{0}/t{1:0>3}/node_{1:0>3}_{0}.pdf'.format(mode,i))
#         plt.savefig(plot_file)
#         plt.close()