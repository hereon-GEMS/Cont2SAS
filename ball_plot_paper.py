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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import ListedColormap

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

xml_folder='./xml/'

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
# element details
el_type='lagrangian'
el_order=1


### sim xml entries ###

# model name
sim_model= 'ball' #root.find('model').text

# simulation parameters
dt= 1. 
t_end= 0.
n_ensem=1
t_arr=np.arange(0,t_end+dt, dt)



### model xml entries ###

# model params
ball_rad=10
ball_sld=2
# dir name for model param
model_param_dir_name = ('rad' + '_' + str(ball_rad) + '_' + 'sld' + '_' + str(ball_sld)).replace('.', 'p')

### scatt_cal xml entries ###

# decreitization params
# number of categories and method of categorization
num_cats= [3, 101, 3] 
method_cats=['simple', 'simple', 'extend']

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
# scatt_settings='cat_' + method_cat + '_' + str(num_cat) + 'Q_' \
#     + str(start_length) + '_' + str(end_length) + '_' + 'orien__' + str(resolution_num)
# scatt_settings=scatt_settings.replace('.', 'p')


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

### save element_order values as strings
el_order_str='lagrangian_' + str(el_order)


mother_dir_name = length_a_str+'_' + length_b_str+'_' + length_c_str\
      + '_' + nx_str + '_' + ny_str+'_' + nz_str + '_' + el_order_str
mother_dir = os.path.join('./data/', mother_dir_name)

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
plot_dir=os.path.join(figure_dir, sim_model +'_method_cat')
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
    # plotting accoring to categorization methods
    fig, ax = plt.subplots(figsize=(7, 
                                    )
    markers=['o', '^', 's']
    colors=['m', 'r', 'k']
    ms_arr=[3, 5, 3]
    for idx in range(len(method_cats)):
        method_cat=method_cats[idx]
        # method_cat_idx=method_cats.index(method_cat)
        num_cat=num_cats[idx]
        print(idx)

        print(method_cat)
        scatt_settings='cat_' + method_cat + '_' + str(num_cat) + 'Q_' \
            + str(start_length) + '_' + str(end_length) + '_' + 'orien_' + '_' + str(resolution_num)
        scatt_settings=scatt_settings.replace('.', 'p')
        # ball geometry
        vol_ball=(4/3)*np.pi*ball_rad**3
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
        ax.loglog(q, Iq,color=colors[idx], linestyle='', marker=markers[idx], 
                  markersize=ms_arr[idx], label='Categorization method: ' + method_cat + ', ' + str(num_cat) + ' categories' )

    # ananlytical intensity 
    ## Intensity unit 10^-10 \AA^2
    Iq_ana,q_ana=ball(qmax=np.max(q),qmin=np.min(q),Npts=100,
                scale=1,bg=0,sld=ball_sld,sld_sol=0,rad=ball_rad)
    ## Normalize by volume
    ## (Before * 10**2) Intensity unit 10^-10 \AA^-1 = 10 ^-2 cm^-1
    ## (after * 10**2) Intensity unit cm^-1
    Iq_ana = (Iq_ana / vol_norm) * 10**2

    # loglog plot
    ax.loglog(q_ana, Iq_ana, 'b', label='Analytical calculation', zorder=-10)
    # ax.loglog(q, Iq,'r', linestyle='', marker='o', markersize=3, label='Numerical calculation')

    # plot formatting
    ## legend
    ax.legend()
    ## labels
    ax.set_xlabel('Q [$\mathrm{\AA}^{-1}$]')
    ax.set_ylabel('I(Q) [$\mathrm{cm}^{-1}$]')
    ## SANS upper boundary Q=1 \AA^-1
    ax.set_xlim(right=1)

    # read node sld
    # save pseudo atom info
    ensem_dir_name='ensem{0:0>3}'.format(0)
    ensem_dir=os.path.join(t_dir, ensem_dir_name)
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

    if el_type=='lagrangian':
        num_node_x=el_order*nx+1
        num_node_y=el_order*ny+1
        num_node_z=el_order*nz+1
    # plotting node SLD
    ## cutting at z = cut_frac * length_z
    cut_frac=0.5
    node_pos_3d=node_pos.reshape(num_node_x, num_node_y, num_node_z, 3)
    z_idx= np.floor(cut_frac*(num_node_z)).astype(int)
    z_val=node_pos_3d[0, 0, z_idx , 2]
    ## figure specification
    plot_file_name='Iq_ball.pdf'
    plot_file=os.path.join(plot_dir,plot_file_name)
    #fig_ins, ax_ins = plt.subplots(figsize=(5, 5))
    ax_ins = inset_axes(ax, width="100%", height="100%", bbox_to_anchor=(0, 0.35, 0.5, 0.5),
                        bbox_transform=ax.transAxes)
    ## image plot
    ### .T is required to exchange x and y axis 
    ### origin is 'lower' to put it in lower left corner 
    node_sld_3d=node_sld.reshape(num_node_x, num_node_y, num_node_z)
    cmap = ListedColormap(['white', '#440154'])
    img = ax_ins.imshow(node_sld_3d[:,:,z_idx].T, 
                    extent=[0, length_a, 0, length_b], 
                    origin='lower', vmin=sld_min, vmax=sld_max, 
                    interpolation='bilinear', cmap=cmap)
    
    ax_ins.axis('off')

    ## add mesh
    cell_x=length_a/nx
    cell_y=length_a/ny
    for idx1 in range(nx):
        for idx2 in range(ny):
            rect_center_x=idx1*cell_x
            rect_center_y=idx2*cell_y
            ax_ins.add_patch(Rectangle((rect_center_x, rect_center_y), cell_x, cell_y, 
                                    edgecolor='k', facecolor='none', linewidth=0.5))

    # # Add arrow with optional text
    # ax.annotate('',
    #             xy=(0.1, 1e6),             # Tip of the arrow
    #             xytext=(0.06, 1e2),         # Start of the arrow (text position)
    #             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
    #             fontsize=10)

    plt.savefig(plot_file, format='pdf')
    plt.close(fig)