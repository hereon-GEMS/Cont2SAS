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

def ball (qmax,qmin,Npts,scale,bg,sld,sld_sol,rad):
    vol=(4/3)*np.pi*rad**3
    del_rho=sld-sld_sol
    q_vec=np.linspace(qmin,qmax,Npts)
    FormFactor=np.zeros(len(q_vec))
    for i in range(len(q_vec)):
        q=q_vec[i]
        if q==0:
            FormFactor[i]=1
        else:
            FormFactor[i]=3*vol*del_rho*J1(q*rad)/(q*rad)
    return ((scale/vol)*np.abs(FormFactor)**2+bg)*10**2, q_vec


#timer counter initial
tic = time.perf_counter()

"""
read input from xml file
"""

### struct xml ###

xml_folder='../xml/'

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
# signal_file=root.find('scatt_cal').find('sig_file').text
scatt_settings='cat_' + method_cat + '_' + str(num_cat) + 'Q_' \
    + str(start_length) + '_' + str(end_length) + '_' + 'orien_' + '_' + str(resolution_num)
scatt_settings=scatt_settings.replace('.', 'p')


"""
read folder structure
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
sim_dir=os.path.join('../data/',
                        length_a_str+'_'+length_b_str+'_'+length_c_str+'_'+
                        nx_str+'_'+ny_str+'_'+nz_str+'/simulation')


# read structure info
data_filename=os.path.join(sim_dir,'../structure/struct.h5')
# nodes, cells, con = dsv.mesh_read(data_filename)

# folder name for model
model_dir_name= (sim_model + '_tend_' + str(t_end) + '_dt_' + str(dt) \
    + '_ensem_' + str(n_ensem)).replace('.','p')
model_dir=os.path.join(sim_dir,model_dir_name)

# folder name for model with particular run param
model_xml=os.path.join(xml_folder, 'model_'+sim_model + '.xml')
tree = ET.parse(model_xml)
root = tree.getroot()
model_param_dir_name=''
for elem in root.iter():
    if elem.text and elem.text.strip():  # Avoid None or empty texts
        model_param_dir_name+= f"{elem.tag}_{elem.text.strip()}_"
model_param_dir_name=model_param_dir_name[0:-1].replace('.', 'p')
model_param_dir=os.path.join(model_dir,model_param_dir_name)

# create folder for figure
## mother folder for figure
figure_dir='../figure/'
os.makedirs(figure_dir, exist_ok=True)
## folder for this suit of figures
plot_dir=os.path.join(figure_dir, 'ball')
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
    # Iq_all_ensem=np.zeros((num_points,n_ensem))
    # q_all_ensem=np.zeros((num_points,n_ensem))
    for j in range(n_ensem):
        idx_ensem=j
        # create ensemble dir
        ensem_dir_name='ensem{0:0>3}'.format(idx_ensem)
        ensem_dir=os.path.join(t_dir, ensem_dir_name)

        # create scatt dir
        scatt_dir_name='scatt_cal_'+ scatt_settings
        scatt_dir=os.path.join(ensem_dir, scatt_dir_name)


        """
        read scatt_cal
        """
        # read node sld
        # save pseudo atom info
        scatt_cal_data_file_name='scatt_cal.h5'
        scatt_cal_data_file=os.path.join(ensem_dir, scatt_cal_data_file_name)
        scatt_cal_data=h5py.File(scatt_cal_data_file,'r')
        node_pos=scatt_cal_data['node_pos'][:]
        node_sld=scatt_cal_data['node_sld'][:]
        pseudo_pos=scatt_cal_data['pseudo_pos'][:]
        pseudo_b=scatt_cal_data['pseudo_b'][:]
        pseudo_b_cat_val=scatt_cal_data['pseudo_b_cat_val'][:]
        pseudo_b_cat_idx=scatt_cal_data['pseudo_b_cat_idx'][:]
        scatt_cal_data.close()

        sld_min=np.min(node_sld,0)
        sld_max=np.max(node_sld,0)


        if idx_ensem==0:
            print('plotting for the particular ensemble')
            # plotting node SLD
            ## cutting at z = cut_frac * length_z
            cut_frac=0.5
            node_pos_3d=node_pos.reshape(nx+1, ny+1, nz+1, 3)
            z_idx= np.floor(cut_frac*(nz+1)).astype(int)
            z_val=node_pos_3d[0, 0, z_idx , 2]
            ## figure specification
            plot_file_name='SLD_ball'
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
            plot_file_name='pseudo_ball'
            plot_file=os.path.join(plot_dir,plot_file_name)
            fig, ax = plt.subplots(figsize=(5, 5))
            ## image plot
            ### .T is required to exchange x and y axis 
            ### origin is 'lower' to put it in lower left corner 
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
            plot_file_name='pseudo_cat_ball'
            plot_file=os.path.join(plot_dir,plot_file_name)
            fig, ax = plt.subplots(figsize=(5, 5))
            ## image plot
            ### .T is required to exchange x and y axis 
            ### origin is 'lower' to put it in lower left corner 
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
            ax.set_xticks([0, 10, 20, 30, 40])
            [(x, x**2) for x in range(5)]

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
    # read radius and sld
    model_xml=os.path.join(xml_folder, 'model_'+sim_model + '.xml')
    tree = ET.parse(model_xml)
    root = tree.getroot()
    ball_rad=int(root.find('rad').text)
    ball_sld=float(root.find('sld').text)
    vol_ball=(4/3)*np.pi*ball_rad**3

    # read signal file
    Iq_data_file_name='Iq_{0}.h5'.format(scatt_settings) 
    Iq_data_file=os.path.join(t_dir,Iq_data_file_name)
    Iq_data=h5py.File(Iq_data_file,'r')
    Iq=Iq_data['Iq'][:]
    Iq=Iq/vol_ball*10**2 # conversion from fm2 to cm-1
    q=Iq_data['Q'][:]
    Iq_data.close()

    
    Iq_ana,q_ana=ball(qmax=np.max(q),qmin=np.min(q),Npts=100,
                scale=1,bg=0,sld=ball_sld,sld_sol=0,rad=ball_rad)

    plot_file_name='Iq_ball'
    plot_file=os.path.join(plot_dir,plot_file_name)
    fig, ax = plt.subplots(figsize=(7, 5))
    ## image plot
    ### .T is required to exchange x and y axis 
    ### origin is 'lower' to put it in lower left corner 
    ax.loglog(q_ana, Iq_ana, 'b')
    ax.loglog(q, Iq,'r', linestyle='', marker='o', markersize=3)
    
    ## plot title
    # title_text=" Cut at Z = {0} {1}".format(z_val_pseudo, r"$\mathrm{\AA}$")
    # ax.set_title(title_text)
    # labels
    ax.set_xlabel('Q [$\mathrm{\AA}^{-1}$]')
    ax.set_ylabel('I(Q) [$\mathrm{cm}^{-1}$]')
    # other formatting
    # ax.set_aspect('equal')
    ax.set_xlim(right=1)
    # ax.set_ylim([0, length_b])
    plt.savefig(plot_file, format='pdf')
    plt.close(fig)
    
    # plt.show()