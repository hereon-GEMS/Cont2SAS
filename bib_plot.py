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

def J1(x):
    if x==0:
        return 0
    else:
        return (np.sin(x)-x*np.cos(x))/x**2
    
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

def ball_in_box(qmax,qmin,Npts,scale,scale2,bg,sld_box, sld_ball,sld_sol,length_a,length_b,length_c,radius):
    length_a, length_b, length_c=arrange_order(length_a,length_b,length_c)
    vol_box=length_a*length_b*length_c
    vol_ball=(4/3)*np.pi*radius**3
    # SLD unit 10^-5 \AA^-2
    del_rho_box=sld_box-sld_sol
    del_rho_ball=sld_ball-sld_sol
    q_arr=np.linspace(qmin,qmax,Npts) 
    Aq_arr=np.zeros(len(q_arr))
    for i in range(len(q_arr)):
        q=q_arr[i]
        if q==0:
            func=lambda alpha, psi:\
                (del_rho_box*vol_box*(np.sinc((1/np.pi)*q*length_a/2*np.sin(alpha)*np.sin(psi)))*\
                (np.sinc((1/np.pi)*q*length_b/2*np.sin(alpha)*np.cos(psi)))*\
                (np.sinc((1/np.pi)*q*length_c/2*np.cos(alpha)))-\
                scale2*1+\
                scale2*1)**2*\
                np.sin(alpha)
        else:
            func=lambda alpha, psi:\
                (del_rho_box*vol_box*(np.sinc((1/np.pi)*q*length_a/2*np.sin(alpha)*np.sin(psi)))*\
                (np.sinc((1/np.pi)*q*length_b/2*np.sin(alpha)*np.cos(psi)))*\
                (np.sinc((1/np.pi)*q*length_c/2*np.cos(alpha)))-\
                scale2*3*vol_ball*del_rho_box*J1(q*radius)/(q*radius)+\
                scale2*3*vol_ball*del_rho_ball*J1(q*radius)/(q*radius))**2*\
                np.sin(alpha)
        
        psi_lim=np.pi
        alpha_lim=np.pi/2 
        # Amplitude unit (\AA^3 * 10^-5 \AA^-2)^2 = 10^-10 \AA^2
        Aq_arr[i]=(1/psi_lim)*gauss_legendre_double_integrate(func,[0, alpha_lim],[0, psi_lim],76)
    Iq_arr = scale*Aq_arr + bg # Intensity unit 10^-10 \AA^2
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
sim_model='bib'
dt=1.
t_end=0.
n_ensem=1

### model_param ###
rad=15
sld_in=2
sld_out=1

### scatt_cal ###
num_cat=101
method_cat='extend'
sig_file='signal.h5'
scan_vec=np.array([1, 0, 0])
Q_range=np.array([0., 1.])
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
plot_dir=os.path.join(figure_dir, sim_model)
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
            if el_type=='lagrangian':
                num_node_x=el_order*nx+1
                num_node_y=el_order*ny+1
                num_node_z=el_order*nz+1
            # plotting node SLD
            ## cutting at z = cut_frac * length_z
            cut_frac=0.5
            node_pos_3d=node_pos.reshape(num_node_x, num_node_y, num_node_z, 3)
            z_idx= np.floor(cut_frac*(nz+1)).astype(int)
            z_val=node_pos_3d[0, 0, z_idx , 2]
            ## figure specification
            plot_file_name='SLD_bib.pdf'
            plot_file=os.path.join(plot_dir,plot_file_name)
            fig, ax = plt.subplots(figsize=(5, 5))
            ## image plot
            ### .T is required to exchange x and y axis 
            ### origin is 'lower' to put it in lower left corner 
            node_sld_3d=node_sld.reshape(nx+1, ny+1, nz+1)
            img = ax.imshow(node_sld_3d[:,:,z_idx].T, 
                            extent=[0, length_a, 0, length_b], 
                            origin='lower', vmin=sld_min, vmax=sld_max, interpolation='bilinear')
            ## color bar
            cbar = plt.colorbar(img, ax=ax)  # Add colorbar to subplot 1
            cbar_label="Scattering length density (SLD) [$10^{-5} \cdot \mathrm{\AA}^{-2}$]"
            cbar.set_label(cbar_label, labelpad=10)
            ## plot title
            title_text=" Cut at Z = {0} {1}".format(z_val, r"$\mathrm{\AA}$")
            ax.set_title(title_text)
            # labels
            ax.set_xlabel(r'X [$\mathrm{\AA}$]')
            ax.set_ylabel(r'Y [$\mathrm{\AA}$]')
            ## add mesh
            cell_x=length_a/nx
            cell_y=length_a/ny
            for idx1 in range(nx):
                for idx2 in range(ny):
                    rect_center_x=idx1*cell_x
                    rect_center_y=idx2*cell_y
                    ax.add_patch(Rectangle((rect_center_x, rect_center_y), cell_x, cell_y, 
                                           edgecolor='k', facecolor='none', linewidth=0.5))
            
            plt.savefig(plot_file, format='pdf')
            plt.close(fig)

            # plotting pseudo atoms
            ## cutting at z = z_val + length_z
            # cell_z= length_c/nz
            # z_val_pseudo=z_val+cell_z
            cut_frac=0.5
            pseudo_pos_3d=pseudo_pos.reshape(nx, ny, nz, 3)
            pseudo_b_3d=pseudo_b.reshape(nx, ny, nz)
            z_idx_pseudo= z_idx-1
            z_val_pseudo=pseudo_pos_3d[0, 0, z_idx_pseudo , 2]
            ## figure specification
            plot_file_name='pseudo_bib.pdf'
            plot_file=os.path.join(plot_dir,plot_file_name)
            fig, ax = plt.subplots(figsize=(5, 5))
            ## scatter plot
            pseudo_pos_x=pseudo_pos_3d[:,:,z_idx_pseudo,0]
            pseudo_pos_y=pseudo_pos_3d[:,:,z_idx_pseudo,1]
            pseudo_b_xy=pseudo_b_3d[:,:,z_idx_pseudo]
            img = ax.scatter(pseudo_pos_x, pseudo_pos_y, c=pseudo_b_xy, s=3, cmap='viridis')
            ## color bar
            cbar = plt.colorbar(img, ax=ax)  # Add colorbar to subplot 1
            cbar_label="Scattering length (b) [$10^{-5} \cdot \mathrm{\AA} = \mathrm{fm}$]"
            cbar.set_label(cbar_label, labelpad=10)
            ## plot title
            title_text=" Cut at Z = {0} {1}".format(z_val_pseudo, r"$\mathrm{\AA}$")
            ax.set_title(title_text)
            # labels
            ax.set_xlabel('X [$\mathrm{\AA}$]')
            ax.set_ylabel('Y [$\mathrm{\AA}$]')
            # other formatting
            ax.set_aspect('equal')
            ax.set_xlim([0, length_a])
            ax.set_ylim([0, length_b])
            ## add mesh
            cell_x=length_a/nx
            cell_y=length_a/ny
            for idx1 in range(nx):
                for idx2 in range(ny):
                    rect_center_x=idx1*cell_x
                    rect_center_y=idx2*cell_y
                    ax.add_patch(Rectangle((rect_center_x, rect_center_y), cell_x, cell_y, 
                                           edgecolor='k', facecolor='none', linewidth=0.5))
            plt.savefig(plot_file, format='pdf')
            plt.close(fig)

            # plotting categorized pseudo atoms
            ## figure specification
            plot_file_name='pseudo_cat_bib.pdf'
            plot_file=os.path.join(plot_dir,plot_file_name)
            fig, ax = plt.subplots(figsize=(5, 5))
            ## scatter plot
            pseudo_b_cat_val_3d=pseudo_b_cat_val.reshape(nx, ny, nz)
            pseudo_b_cat_val_xy=pseudo_b_cat_val_3d[:,:,z_idx_pseudo]
            img = ax.scatter(pseudo_pos_x, pseudo_pos_y, c=pseudo_b_cat_val_xy, s=3, cmap='viridis')
            ## color bar
            cbar = plt.colorbar(img, ax=ax)  # Add colorbar to subplot 1
            cbar_label="Scattering length (b) [$10^{-5} \cdot \mathrm{\AA} = \mathrm{fm}$]"
            cbar.set_label(cbar_label, labelpad=10)
            ## plot title
            title_text=" Cut at Z = {0} {1}".format(z_val_pseudo, r"$\mathrm{\AA}$")
            ax.set_title(title_text)
            # labels
            ax.set_xlabel('X [$\mathrm{\AA}$]')
            ax.set_ylabel('Y [$\mathrm{\AA}$]')
            # other formatting
            ax.set_aspect('equal')
            ax.set_xlim([0, length_a])
            ax.set_ylim([0, length_b])
            x_ticks_val=np.round(np.linspace(0, length_a, 5),2)
            ax.set_xticks(list(x_ticks_val))
            y_ticks_val=np.round(np.linspace(0, length_b, 5),2)
            ax.set_yticks(list(y_ticks_val))
            ## add mesh
            cell_x=length_a/nx
            cell_y=length_a/ny
            for idx1 in range(nx):
                for idx2 in range(ny):
                    rect_center_x=idx1*cell_x
                    rect_center_y=idx2*cell_y
                    ax.add_patch(Rectangle((rect_center_x, rect_center_y), cell_x, cell_y, 
                                           edgecolor='k', facecolor='none', linewidth=0.5))
            plt.savefig(plot_file, format='pdf')
            plt.close(fig)
    
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

    # ananlytical intensity 
    ## Intensity unit 10^-10 \AA^2
    Iq_ana,q_ana= ball_in_box(qmax=np.max(q),qmin=np.min(q),Npts=100,
                scale=1,scale2=1,bg=0,sld_box=sld_out, sld_ball=sld_in, sld_sol=0,
                length_a=length_a, length_b=length_b, length_c=length_c, radius=rad)
    ## Normalize by volume
    ## (Before * 10**2) Intensity unit 10^-10 \AA^-1 = 10 ^-2 cm^-1
    ## (after * 10**2) Intensity unit cm^-1
    Iq_ana = (Iq_ana / vol_norm) * 10**2
    plot_file_name='Iq_bib.pdf'
    plot_file=os.path.join(plot_dir,plot_file_name)
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # loglog plot
    ax.loglog(q_ana, Iq_ana, 'b', label= 'Analytical calculation')
    ax.loglog(q, Iq,'r', linestyle='', marker='o', markersize=3, label= 'Numerical calculation')
    
    # plot formatting
    ## legend
    ax.legend()
    ## labels
    ax.set_xlabel('Q [$\mathrm{\AA}^{-1}$]')
    ax.set_ylabel('I(Q) [$\mathrm{cm}^{-1}$]')
    ## SANS upper boundary Q=1 \AA^-1
    ax.set_xlim(right=1)
    ## save plot
    plt.savefig(plot_file, format='pdf')
    plt.close(fig)