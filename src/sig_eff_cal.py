#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
from lib import sig_eff as sigeff



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
import matplotlib.patches as patches


#timer counter initial
tic = time.perf_counter()

"""
read input from xml file
"""

### struct xml ###

xml_folder='./xml/'

struct_xml=os.path.join(xml_folder, 'struct.xml')

tree=ET.parse(struct_xml)
root = tree.getroot()

# box side lengths
length_a=float(root.find('lengths').find('x').text) 
length_b=float(root.find('lengths').find('y').text)
length_c=float(root.find('lengths').find('z').text)
# number of cells in each direction
nx=int(root.find('num_cell').find('x').text)
ny=int(root.find('num_cell').find('y').text)
nz=int(root.find('num_cell').find('z').text)
# mid point of structure
mid_point=np.array([length_a/2, length_b/2, length_c/2])
# element type
el_type=root.find('element').find('type').text
if el_type=='lagrangian':
    el_order=int(root.find('element').find('order').text)
    el_info={'type': el_type,
             'order': el_order}

### sim xml ###

sim_xml=os.path.join(xml_folder, 'simulation.xml')

tree=ET.parse(sim_xml)
root = tree.getroot()

# model name
sim_model=root.find('model').text

# simulation parameters
## time
dt=float(root.find('sim_param').find('dt').text)
t_end=float(root.find('sim_param').find('tend').text)
t_arr=np.arange(0,t_end+dt, dt)
## ensemble
n_ensem=int(root.find('sim_param').find('n_ensem').text)


### scatt_cal xml ###
# scatter calculation
# scatt_cal
scatt_cal_xml=os.path.join(xml_folder, 'scatt_cal.xml')

tree=ET.parse(scatt_cal_xml)
root = tree.getroot()

# decreitization params
# number of categories and method of categorization
num_cat=int(root.find('discretization').find('num_cat').text)
method_cat=root.find('discretization').find('method_cat').text

# scatt_cal params
signal_file=root.find('scatt_cal').find('sig_file').text
resolution_num=int(root.find('scatt_cal').find('num_orientation').text)
start_length=float(root.find('scatt_cal').find('Q_start').text)
end_length=float(root.find('scatt_cal').find('Q_end').text)
num_points=int(root.find('scatt_cal').find('num_points').text)
scan_vec_x=float(root.find('scatt_cal').find('scan_vec').find('x').text)
scan_vec_y=float(root.find('scatt_cal').find('scan_vec').find('y').text)
scan_vec_z=float(root.find('scatt_cal').find('scan_vec').find('z').text)
scan_vector=[scan_vec_x, scan_vec_y, scan_vec_z]
scatt_settings='cat_' + method_cat + '_' + str(num_cat) + 'Q_' \
    + str(start_length) + '_' + str(end_length) + '_' + 'orien_' + '_' + str(resolution_num)
scatt_settings=scatt_settings.replace('.', 'p')

### sigeff xml ###
sig_eff_xml=os.path.join(xml_folder, 'sig_eff.xml')

tree=ET.parse(sig_eff_xml)
root = tree.getroot()

# decreitization params
# number of categories and method of categorization
instrument=root.find('instrument').text
facility=root.find('facility').text

# scatt_cal params
d=float(root.find('d').text)
wl=float(root.find('lambda').text) # wavelength
beam_shift_vec_x=float(root.find('beam_center').find('x').text)
beam_shift_vec_y=float(root.find('beam_center').find('y').text)
#beam_center_vec_z=float(root.find('beam_center').find('z').text)
beam_shift_vector=np.array([beam_shift_vec_x, beam_shift_vec_y])

"""
create folder structure, read structure and sld info
"""

# folder structure
## mother folder for simulation 
### save length values as strings
### decimal points are replaced with p
length_a_str=str(length_a).replace('.','p')
length_b_str=str(length_a).replace('.','p')
length_c_str=str(length_a).replace('.','p')

### save num_cell values as strings
nx_str=str(nx)
ny_str=str(ny)
nz_str=str(nz)
struct_folder_name = (length_a_str + '_' + length_b_str + '_' + length_c_str
                       + '_' + nx_str + '_' + ny_str + '_' + nz_str)

# save elemnt type as string
if el_type=='lagrangian':
    el_order_str=str(el_order)
    struct_folder_name += '_' + el_type + '_' + el_order_str


sim_dir=os.path.join('./data/', struct_folder_name +'/simulation')
# sim_dir=os.path.join('../data/',
#                         length_a_str+'_'+length_b_str+'_'+length_c_str+'_'+
#                         nx_str+'_'+ny_str+'_'+nz_str+'/simulation')
os.makedirs(sim_dir, exist_ok=True)

# read structure info
data_filename=os.path.join(sim_dir,'../structure/struct.h5')
nodes, cells, con = dsv.mesh_read(data_filename)

# folder name for model
model_dir_name= (sim_model + '_tend_' + str(t_end) + '_dt_' + str(dt) \
    + '_ensem_' + str(n_ensem)).replace('.','p')
model_dir=os.path.join(sim_dir,model_dir_name)

# folder name for run of model with particular run param
model_xml=os.path.join(xml_folder, 'model_'+sim_model + '.xml')
tree = ET.parse(model_xml)
root = tree.getroot()
model_param_dir_name=''
for elem in root.iter():
    if elem.text and elem.text.strip():  # Avoid None or empty texts
        model_param_dir_name+= f"{elem.tag}_{elem.text.strip()}_"
model_param_dir_name=model_param_dir_name[0:-1]
model_param_dir=os.path.join(model_dir,model_param_dir_name)

if os.path.exists(model_param_dir):
    print('calculating effective cross-section')
else:
    print('create simulation first')

## read detector geometry
detector_file_name='detector.h5'
detector_dir='./detector_geometry/{0}_{1}'.format(instrument, facility)
detector_file=os.path.join(detector_dir, detector_file_name)
detector_file_data=h5py.File(detector_file,'r')
nx_det=detector_file_data['num_pixel_x'][()]
ny_det=detector_file_data['num_pixel_y'][()]
dx_det=detector_file_data['width_pixel_x'][()]
dy_det=detector_file_data['width_pixel_y'][()]
pixel_coord_det=detector_file_data['pixel_coord'][:]
bs_wx_det=detector_file_data['beam_stop_width_x'][()]
bs_wy_det=detector_file_data['beam_stop_width_y'][()]
detector_file_data.close()
# calculate beam center
org_det=np.array([nx_det*dx_det/2, ny_det*dy_det/2])
org_beam=org_det+beam_shift_vector
# calculate r from pixel coordinates
pixel_r=np.sqrt(np.sum((pixel_coord_det-org_beam)**2, axis=1))
pixel_Q=(4*np.pi/wl)*np.sin(0.5*np.arctan(pixel_r/d))
Q_range=[min(pixel_Q), max(pixel_Q)]

# initialize sigma_eff
sig_eff=np.zeros(len(t_arr))
for i in range(len(t_arr)):
    t=t_arr[i]
    # time_dir name
    t_dir_name='t{0:0>3}'.format(i)
    t_dir=os.path.join(model_param_dir, t_dir_name)
    # read I and Q
    Iq_data_file_name='Iq_{0}.h5'.format(scatt_settings)
    Iq_data_file=os.path.join(t_dir,Iq_data_file_name)
    Iq_data=h5py.File(Iq_data_file,'r')
    Iq=Iq_data['Iq'][:]
    q=Iq_data['Q'][:]
    Iq_data.close()
    # q cut
    q_cut = q[(q >= Q_range[0]) & (q <= Q_range[1])]
    Iq_cut= Iq[(q >= Q_range[0]) & (q <= Q_range[1])]
    # # Uncomment till plt.show() if you want I vs Q to appear
    # # plotitng I vs Q in time folder
    # plt.loglog(q_cut,Iq_cut)
    # plt.xlabel('Q')
    # plt.ylabel('I(Q)')
    # plt.show()

    # calculate weight
    # categorize Q pixel and weight calculation
    q_cat=np.zeros_like(pixel_Q)
    w_cat=np.zeros_like(pixel_Q)
    wq=np.zeros_like(q_cut)
    for q_cat_idx in range(len(q_cut)):
        q_cat_val=q_cut[q_cat_idx]
        if q_cat_idx == 0:
            cat_low_bound=Q_range[0]
        else:
            cat_low_bound=(q_cut[q_cat_idx]+q_cut[q_cat_idx-1])/2
        if q_cat_idx == len(q_cut)-1:
            cat_upper_bound=Q_range[1]
        else:
            cat_upper_bound=(q_cut[q_cat_idx]+q_cut[q_cat_idx+1])/2
        q_cat[(pixel_Q >= cat_low_bound) & (pixel_Q <= cat_upper_bound)]=q_cat_val
        wq[q_cat_idx]=len(pixel_Q[(pixel_Q >= cat_low_bound) & (pixel_Q <= cat_upper_bound)])
        w_cat[(pixel_Q >= cat_low_bound) & (pixel_Q <= cat_upper_bound)]=wq[q_cat_idx]
        
    # plotitng I vs Q in time folder
    plt.scatter(pixel_coord_det[:,0], pixel_coord_det[:,1], c=w_cat, cmap='viridis_r', s=1)
    plt.axis('equal')
    plt.xlabel('x [$m$]')
    plt.ylabel('y [$m$]')
    plt.colorbar(label='categorized q values')
    beam_stop = patches.Rectangle(((nx_det*dx_det-bs_wx_det)/2, (nx_det*dy_det-bs_wy_det)/2),
                                   0.085, 0.085, color='orange', fill=True)
    plt.gca().add_patch(beam_stop)
    outercircle_out = patches.Circle(((nx_det*dx_det)/2, (nx_det*dy_det)/2), 
                                     np.sqrt((nx_det*dx_det/2)**2+(ny_det*dy_det/2)**2), 
                                     color='r', fill=False)
    outercircle_in = patches.Circle(((nx_det*dx_det)/2, (nx_det*dy_det)/2), 
                                     (nx_det*dx_det/2), 
                                     color='r', fill=False)
    plt.gca().add_patch(outercircle_out)
    plt.gca().add_patch(outercircle_in)
    innercircle_out = patches.Circle(((nx_det*dx_det)/2, (nx_det*dy_det)/2), 
                                     np.sqrt((bs_wx_det/2)**2+(bs_wy_det/2)**2), 
                                     color='k', fill=False)
    innercircle_in = patches.Circle(((nx_det*dx_det)/2, (nx_det*dy_det)/2), 
                                     (bs_wx_det/2), 
                                     color='k', fill=False)
    plt.gca().add_patch(innercircle_out)
    plt.gca().add_patch(innercircle_in)
    fig_org=[(nx_det*dx_det)/2, (nx_det*dy_det)/2]
    fig_diag_len=np.sqrt((nx_det*dx_det)**2+(ny_det*dy_det)**2)
    fig_pad=0.05*fig_diag_len/2
    plt.xlim([fig_org[0]-fig_diag_len/2-fig_pad, fig_org[0]+fig_diag_len/2+fig_pad])
    plt.ylim([fig_org[0]-fig_diag_len/2-fig_pad, fig_org[0]+fig_diag_len/2+fig_pad])
    wq_plot_file_name='weightvsq.jpg'
    wq_plot_file = os.path.join(t_dir, wq_plot_file_name)
    plt.savefig(wq_plot_file, format='jpg')
    plt.show()

    # combine intensity and weights
    Iq_total=Iq_cut*wq

    # calculate sigma eff
    
    for j in range(len(q_cut)-1):
        del_q=q_cut[j+1]-q_cut[j]
        sig_eff[i]+=0.5*del_q*(Iq_total[j+1]+Iq_total[j])
sig_eff_data_file_name='sig_eff_{0}.h5'.format(scatt_settings)
sig_eff_data_file=os.path.join(model_param_dir,sig_eff_data_file_name)
sig_eff_data=h5py.File(sig_eff_data_file,'w')
sig_eff_data['sig_eff']=sig_eff
sig_eff_data['t']=t_arr
sig_eff_data.close()

plt.plot(t_arr, sig_eff)
sig_eff_plot_file_name='sig_eff_{0}.jpg'.format(scatt_settings)
sig_eff_plot_file = os.path.join(model_param_dir, sig_eff_plot_file_name)
plt.savefig(sig_eff_plot_file, format='jpg')
plt.show()