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

def arrange_order(a,b,c):
    zusammen=[a,b,c]
    zusammen.sort()
    return zusammen[0], zusammen[1], zusammen[2]

def gauss_legendre_double_integrate(func, domain1, domain2, deg):
    x, w = np.polynomial.legendre.leggauss(deg)
    xgrid, ygrid=np.meshgrid(x,x)
    x=xgrid.reshape((1,np.size(xgrid)))
    y=ygrid.reshape((1,np.size(ygrid)))
    wx, wy=np.meshgrid(w,w)
    w1=wx.reshape((1,np.size(wx)))
    w2=wy.reshape((1,np.size(wy)))
    s1 = (domain1[1] - domain1[0])/2
    a1 = (domain1[1] + domain1[0])/2
    s2 = (domain2[1] - domain2[0])/2
    a2 = (domain2[1] + domain2[0])/2
    return np.sum(s1*s2*w1*w2*func(s1*x + a1,s2*y + a2))

def box (qmax,qmin,Npts,scale,bg,sld,sld_sol,length_a,length_b,length_c):
    length_a, length_b, length_c= arrange_order(length_a,length_b,length_c)
    vol_box=length_a*length_b*length_c
    # SLD unit 10^-5 \AA^-2
    del_rho_box=sld-sld_sol
    q_arr=np.linspace(qmin,qmax,Npts) 
    Aq_arr=np.zeros(len(q_arr))
    for i in range(len(q_arr)):
        q=q_arr[i]
        func=lambda alpha, psi:\
            (del_rho_box*vol_box*(np.sinc((1/np.pi)*q*length_a/2*np.sin(alpha)*np.sin(psi)))*\
            (np.sinc((1/np.pi)*q*length_b/2*np.sin(alpha)*np.cos(psi)))*\
            (np.sinc((1/np.pi)*q*length_c/2*np.cos(alpha))))**2*\
            np.sin(alpha)
            
        psi_lim=np.pi
        alpha_lim=np.pi/2 
        # Amplitude unit (\AA^3 * 10^-5 \AA^-2)^2 = 10^-10 \AA^2
        Aq_arr[i]=(1/psi_lim)*gauss_legendre_double_integrate(func,[0, alpha_lim],[0, psi_lim],76)
    Iq_arr = scale*Aq_arr # Intensity unit 10^-10 \AA^2
    return Iq_arr, q_arr

"""
Input data
"""
# script dir and working dir
script_dir = "src"  # Full path of the script
working_dir = "."  # Directory to run the script in


### struct gen ###
xml_dir=os.path.join(working_dir, './xml') 
length_a=40. 
length_b=40. 
length_c=40.
nx=40 
ny=40 
nz=40 
el_type='lagrangian'
el_order=1
update_val=True
plt_node=False
plt_cell=False
plt_mesh=False

### sim_gen ###
sim_model='box'
dt=1.
t_end=0.
n_ensem=1

### model_param ###
sld=2

### scatt_cal ###
num_cat=3
method_cat='extend'
sig_file='signal.h5'
scan_vec=np.array([1, 0, 0])
Q_range=np.array([0., 1.])
num_points=100
num_orientation_arr=[10, 50, 200]

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
model_param_dir_name = ('sld' + '_' + str(sld)).replace('.', 'p')

# folder name for model with particular run param
model_param_dir=os.path.join(model_dir,model_param_dir_name)


### scatt_cal xml entries ###

# scatt_cal params
start_length=Q_range[0]
end_length=Q_range[1]

# dir name
# scatt_settings to to be defined later


"""
read folder structure
"""

# create folder for figure (one level up from data folder)
figure_dir=os.path.join(mother_dir, '../../figure/')
os.makedirs(figure_dir, exist_ok=True)
## folder for this suit of figures
plot_dir=os.path.join(figure_dir, sim_model + '_orien')
os.makedirs(plot_dir, exist_ok=True)

if os.path.exists(model_param_dir):
    print('model folder exists')
else:
    print('model folder does not exist')

for i in range(len(t_arr)):
    t=t_arr[i]
    # time_dir name
    t_dir_name='t{0:0>3}'.format(i)
    t_dir=os.path.join(model_param_dir, t_dir_name)
    # plotting accoring to orientations
    fig, ax = plt.subplots(figsize=(7, 5))
    markers=['^', 's', 'o']
    colors=['g', 'b', 'r']
    ms_arr=[3, 3, 3]
    for num_orientation in num_orientation_arr:
        num_orien_idx=num_orientation_arr.index(num_orientation)
        scatt_settings='cat_' + method_cat + '_' + str(num_cat) + 'Q_' \
            + str(start_length) + '_' + str(end_length) + '_' + 'orien_' + '_' + str(num_orientation)
        scatt_settings=scatt_settings.replace('.', 'p')

        # box geometry
        vol_box=length_a*length_b*length_c

        # volume for normalization
        vol_norm=vol_box

        # numerical intensity
        ## read I vs Q signal file
        Iq_data_file_name='Iq_{0}.h5'.format(scatt_settings) 
        Iq_data_file=os.path.join(t_dir,Iq_data_file_name)
        Iq_data=h5py.File(Iq_data_file,'r')
        Iq=Iq_data['Iq'][:] # unit fm^2
        Iq=Iq/vol_norm*10**2 # unit (fm^2 / \AA^3) * 10^2 = cm^-1
        q=Iq_data['Q'][:]
        Iq_data.close()
    
        # loglog plot
        ax.loglog(q, Iq,color=colors[num_orien_idx], linestyle='', 
                  marker=markers[num_orien_idx], markersize=ms_arr[num_orien_idx],
                   label= 'Number of oreientations = {0}'.format(num_orientation))

    # ananlytical intensity 
    ## Intensity unit 10^-10 \AA^2
    Iq_ana,q_ana= box(qmax=np.max(q),qmin=np.min(q),Npts=100,
                scale=1,bg=0,sld=sld,sld_sol=0,
                length_a=length_a, length_b=length_b, length_c=length_c)
    ## Normalize by volume
    ## (Before * 10**2) Intensity unit 10^-10 \AA^-1 = 10 ^-2 cm^-1
    ## (after * 10**2) Intensity unit cm^-1
    Iq_ana = (Iq_ana / vol_norm) * 10**2
    
    ax.loglog(q_ana, Iq_ana, 'gray', label= 'Analytical calculation')

    # plot formatting
    ## legend
    ax.legend()
    ## labels
    ax.set_xlabel('Q [$\mathrm{\AA}^{-1}$]')
    ax.set_ylabel('I(Q) [$\mathrm{cm}^{-1}$]')
    ## SANS upper boundary Q=1 \AA^-1
    ax.set_xlim(right=1)
    ## save plot
    plot_file_name='Iq_box_orien.pdf'
    plot_file=os.path.join(plot_dir,plot_file_name)
    plt.savefig(plot_file, format='pdf')
    plt.close(fig)