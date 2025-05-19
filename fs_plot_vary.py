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
from lib import fitter as fit



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
from scipy.optimize import curve_fit

def J1(x):
    if x==0:
        return 0
    else:
        return (np.sin(x)-x*np.cos(x))/x**2

def fuzzysph(qmax,qmin,Npts,scale,bg,sld,sld_sol,sig_fuzz,radius):
    vol=(4/3)*np.pi*radius**3
    # SLD unit 10^-5 \AA^-2
    del_rho=sld-sld_sol
    q_arr=np.linspace(qmin,qmax,Npts)
    FormFactor=np.zeros(len(q_arr))
    for i in range(len(q_arr)):
        q=q_arr[i]
        if q==0:
            # Form factor unit 10^-5 \AA
            FormFactor[i]=vol*del_rho
        else:
            # Form factor unit 10^-5 \AA
            FormFactor[i]=(3*vol*del_rho*J1(q*radius)/(q*radius))*np.exp((-(sig_fuzz*q)**2)/2)
    # Intensity unit 10^-10 \AA^2
    Iq_arr = scale*np.abs(FormFactor)**2+bg
    return Iq_arr, q_arr

def fit_func(q_in, sig_opt, rad_opt):
    Iq, q_out =fuzzysph(qmax=max(q_in),qmin=min(q_in),
                        Npts=len(q_in), scale=1 , bg=0,
                        sld=sld_in, sld_sol=sld_out,
                        sig_fuzz=sig_opt, radius=rad_opt)
    return Iq

"""
Input data
"""
# script dir and working dir
script_dir = "src"  # Full path of the script
working_dir = "."  # Directory to run the script in


nx_arr=[50,100,50]
el_order_arr=[1,1,2]


# plot fit fuzz value or sigma for all time step
# plot_file_name='sig_fit_{0}.pdf'.format(sim_model)
# plot_file=os.path.join(plot_dir,plot_file_name)
fig1, ax1 = plt.subplots(figsize=(7, 5))
fig2, ax2 = plt.subplots(figsize=(7, 5))

marker=['s', '^', 'o']
colors=['lime', 'm', 'yellow']
ms=[6,6,4]

for vary_idx in range(len(nx_arr)):
    ### struct gen ###
    xml_dir=os.path.join(working_dir, './xml') 
    length_a=200. 
    length_b=length_a 
    length_c=length_a
    nx=nx_arr[vary_idx] 
    ny=nx 
    nz=nx 
    el_type='lagrangian'
    el_order=el_order_arr[vary_idx]
    update_val=True
    plt_node=False
    plt_cell=False
    plt_mesh=False

    cell_x=length_a/nx
    cell_y=length_a/nx
    cell_z=length_a/nx

    ### sim_gen ###
    sim_model='fs'
    dt=1.
    t_end=10.
    n_ensem=1

    ### model_param ###
    rad=60
    sig_0=2
    sig_end=10
    sld_in=5
    sld_out=1

    ### scatt_cal ###
    num_cat=501
    method_cat='extend'
    sig_file='signal.h5'
    scan_vec=np.array([1, 0, 0])
    Q_range=np.array([0., 0.2])
    num_points=100
    num_orientation=100

    """
    calculate vars and create folder structure
    """

    xml_folder=xml_dir

    ### struct xml ###

    # calculate mid point of structure (simulation box)
    mid_point=np.array([length_a/2, length_b/2, length_c/2])

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

    ### save element_order values as strings
    el_order_str='lagrangian_' + str(el_order)

    mother_dir_name = length_a_str+'_' + length_b_str+'_' + length_c_str\
        + '_' + nx_str + '_' + ny_str + '_' + nz_str + '_' + el_order_str
    data_dir=os.path.join(working_dir, 'data')
    mother_dir = os.path.join(data_dir, mother_dir_name)

    # read structure info
    data_file=os.path.join(mother_dir, 'structure/struct.h5')

    ### sim xml entries ###

    # time array
    t_arr=np.arange(0,t_end+dt, dt)

    # dir name
    sim_dir=os.path.join(mother_dir, 'simulation')



    ### model xml entries ###
    # folder name for model
    model_dir_name= (sim_model + '_tend_' + str(t_end) + '_dt_' + str(dt) \
        + '_ensem_' + str(n_ensem)).replace('.','p')
    model_dir=os.path.join(sim_dir,model_dir_name)

    # dir name for model param
    model_param_dir_name = ('rad' + '_' + str(rad) + '_' +
                            'sig_0' + '_' + str(sig_0) + '_' +
                            'sig_end' + '_' + str(sig_end) + '_' +
                            'sld_in' + '_' + str(sld_in) + '_' +
                            'sld_out' + '_' + str(sld_out)).replace('.', 'p')

    # folder name for model with particular run param
    model_param_dir=os.path.join(model_dir,model_param_dir_name)


    ### scatt_cal xml entries ###

    # scatt_cal params
    start_length=Q_range[0]
    end_length=Q_range[1]
    num_points=100 #int(root.find('scatt_cal').find('num_points').text)

    # dir name
    scatt_settings='cat_' + method_cat + '_' + str(num_cat) + 'Q_' \
        + str(start_length) + '_' + str(end_length) + '_' + 'orien_' + '_' + str(num_orientation)
    scatt_settings=scatt_settings.replace('.', 'p')


    """
    read folder structure
    """

    # create folder for figure (one level up from data folder)
    figure_dir=os.path.join(mother_dir, '../../figure/')
    os.makedirs(figure_dir, exist_ok=True)
    ## folder for this suit of figures
    plot_dir=os.path.join(figure_dir, sim_model + '_vary')
    os.makedirs(plot_dir, exist_ok=True)

    if os.path.exists(model_param_dir):
        print('model folder exists')
    else:
        print('model folder does not exist')

    # initialize fit_params (radius)
    rad_fit=np.zeros_like(t_arr)
    rad_ana=np.ones_like(t_arr)*rad
    sig_fit=np.zeros_like(t_arr)
    sig_ana=sig_0+t_arr*(sig_end-sig_0)/t_end

    # color scheme
    color_rainbow = plt.cm.rainbow(np.linspace(0, 1, len(t_arr)))

    for i in range(len(t_arr)):
        t=t_arr[i]
        t_str=str(t).replace('.','p')
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

        
        # box geometry
        vol_box=length_a*length_b*length_c

        # volume for normalization
        vol_norm=vol_box

        # numerical intensity
        ## read I vs Q signal file
        Iq_data_file_name='Iq_{0}.h5'.format(scatt_settings) 
        Iq_data_file=os.path.join(t_dir,Iq_data_file_name)
        Iq_data=h5py.File(Iq_data_file,'r')
        Iq_num_raw=Iq_data['Iq'][:] # unit fm^2
        Iq_num=Iq_num_raw/vol_norm*10**2 # unit (fm^2 / \AA^3) * 10^2 = cm^-1
        q_num=Iq_data['Q'][:]
        Iq_data.close()

        # fit radius and sig (fuzz_value) w.r.t. numerical intensity
        popt, pcov = curve_fit(fit_func, q_num, Iq_num_raw, 
                            bounds=([sig_0, rad-cell_x], [sig_end, rad+cell_x]))
        sig_fit[i]=round(np.abs(popt[0]),2)
        rad_fit[i]=round(popt[1],2)

    ax1.plot(t_arr, sig_fit, linestyle='', marker=marker[vary_idx], 
             color=colors[vary_idx], markersize=ms[vary_idx], markeredgecolor='k', markeredgewidth=0.5,
             label= 'Meshing: {0} elements of order {1}'.format(nx*ny*nz, el_order))
    
    ax2.plot(t_arr, rad_fit, linestyle='', marker=marker[vary_idx], 
             color=colors[vary_idx], markersize=ms[vary_idx], markeredgecolor='k', markeredgewidth=0.5,
             label= 'Meshing: {0} elements of order {1}'.format(nx*ny*nz, el_order))


ax1.plot(t_arr, sig_ana, 'gray', zorder=-10, label= 'Simulation value')
# plot formatting
## legend
ax1.legend()
## labels
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Fuzzyness [$\mathrm{\AA}$]')
## limits
#ax1.grid(True)
plot_file_name='sig_fit_{0}.pdf'.format(sim_model)
plot_file=os.path.join(plot_dir,plot_file_name)
fig1.savefig(plot_file, format='pdf')


ax2.plot(t_arr, rad_ana, 'gray', zorder=-10, label= 'Simulation value')
# plot formatting
## legend
ax2.legend(loc='upper left')
## labels
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Radius of grain [$\mathrm{\AA}$]')
## limits
ax2.set_ylim([rad-4,rad+4])
#ax2.grid(True)
plot_file_name='rad_fit_{0}.pdf'.format(sim_model)
plot_file=os.path.join(plot_dir,plot_file_name)
fig2.savefig(plot_file, format='pdf')