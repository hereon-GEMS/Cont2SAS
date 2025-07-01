"""
This assigns SLD to the 3D strcuture and saves it in the data folder

Created on Fri Jun 23 10:28:09 2023

@author: Arnab Majumdar
"""
import sys
import os
import time
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import h5py
import imageio.v2 as imageio


# find current dir and and ..
lib_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# add .. in path
sys.path.append(lib_dir)
# lib imports
from lib import struct_gen as sg # pylint: disable=import-error, wrong-import-position
from lib import simulation as sim # pylint: disable=import-error, wrong-import-position




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


"""
create folder structure and read structure info
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
os.makedirs(sim_dir, exist_ok=True)


# read structure info
data_filename=os.path.join(sim_dir,'../structure/struct.h5')
nodes, cells, con, mesh = sg.mesh_read(data_filename)

# create folder for model
model_dir_name= (sim_model + '_tend_' + str(t_end) + '_dt_' + str(dt) \
    + '_ensem_' + str(n_ensem)).replace('.','p')
model_dir=os.path.join(sim_dir,model_dir_name)
os.makedirs(model_dir, exist_ok=True)

# create folder for run of model with particular run param
model_xml=os.path.join(xml_folder, 'model_'+sim_model + '.xml')
tree = ET.parse(model_xml)
root = tree.getroot()
model_param_dir_name=''
for elem in root.iter():
    if elem.text and elem.text.strip():  # Avoid None or empty texts
        model_param_dir_name+= f"{elem.tag}_{elem.text.strip()}_"
model_param_dir_name=model_param_dir_name[0:-1].replace('.', 'p')
model_param_dir=os.path.join(model_dir,model_param_dir_name)
os.makedirs(model_dir, exist_ok=True)

images_1=[] # cut at z = 1/4 * z_max
images_2=[] # cut at z = 2/4 * z_max (2/4 = 1/2)
images_3=[] # cut at z = 3/4 * z_max
for i, t in enumerate(t_arr):
    # create time_dir
    t_dir_name=f't{i:0>3}'
    t_dir=os.path.join(model_param_dir, t_dir_name)
    os.makedirs(t_dir, exist_ok=True)
    for j in range(n_ensem):
        idx_ensem=j
        # create ensemble dir
        ensem_dir_name=f'ensem{idx_ensem:0>3}'
        ensem_dir=os.path.join(t_dir, ensem_dir_name)
        os.makedirs(ensem_dir, exist_ok=True)
        ################
        # run simulation
        ################
        sld, sld_max, sld_min=sim.model_run(sim_model, nodes, mid_point, t, t_end)

        # number for nodes in x, y, z direction
        if el_type=='lagrangian':
            num_node_x=el_order*nx+1
            num_node_y=el_order*ny+1
            num_node_z=el_order*nz+1
        # plottig
        sld_3d=sld.reshape(num_node_x, num_node_y, num_node_z)
        ## z = 1/4 * z_max
        nodes_3d=nodes.reshape(num_node_x, num_node_y, num_node_z, 3)
        z_val=nodes_3d[0, 0, num_node_z//4, 2]
        ### .T is required to exchange x and y axis
        ### origin is 'lower' to put it in lower left corner
        plt.imshow(sld_3d[:,:,num_node_z//4].T,
                   extent=[0, length_a, 0, length_b],
                   origin='lower',
                   vmin=sld_min, vmax=sld_max)
        plot_file_1=os.path.join(ensem_dir,f'snap_z_{z_val}.jpg')
        plt.colorbar()
        plt.title(f" time = {t:0>3}s \n emsemble step = {idx_ensem+1:0>3} \n z = {z_val}{r'$\AA$'}")
        plt.savefig(plot_file_1, format='jpg', bbox_inches='tight')
        ### add images of ensemble 1 for video
        if idx_ensem==0:
            images_1.append(imageio.imread(plot_file_1))
            if sim_model=='bib_ecc':
                plt.show()
        plt.close()

        ## z = 2/4 * z_max
        nodes_3d=nodes.reshape(num_node_x, num_node_y, num_node_z, 3)
        z_val=nodes_3d[0, 0, (num_node_z)//2, 2]
        ### .T is required to exchange x and y axis
        ### origin is 'lower' to put it in lower left corner
        plt.imshow(sld_3d[:,:,(num_node_z)//2].T,
                    extent=[0, length_a, 0, length_b],
                      origin='lower',
                        vmin=sld_min, vmax=sld_max)
        plot_file_2=os.path.join(ensem_dir,f'snap_z_{z_val}.jpg')
        plt.colorbar()
        plt.title(f"time = {t:0>3} s, ensmbl num = {idx_ensem+1}, z = {z_val}{r'$\mathrm{\AA}$'}")
        #plt.title(r"time = {0:0>3}s, {1}".format(t,r'$\mathrm{\AA}$'))
        plt.savefig(plot_file_2, format='jpg')
        ### add images of ensemble 1 for video
        if idx_ensem==0:
            images_2.append(imageio.imread(plot_file_2))
            plt.show()
        plt.close()

        ## z = 3/4 * z_max
        nodes_3d=nodes.reshape(num_node_x, num_node_y, num_node_z, 3)
        z_val=nodes_3d[0, 0, 3*(num_node_z)//4, 2]
        ### .T is required to exchange x and y axis
        ### origin is 'lower' to put it in lower left corner
        plt.imshow(sld_3d[:,:,3*(num_node_z)//4].T,
                    extent=[0, length_a, 0, length_b],
                      origin='lower',
                        vmin=sld_min, vmax=sld_max)
        plot_file_3=os.path.join(ensem_dir,'snap_z_{z_val}.jpg')
        plt.colorbar()
        plt.title(f" time = {t:0>3}s, emsemble step = {idx_ensem+1:0>3} \n z = {z_val}{r'$\AA$'}")
        plt.savefig(plot_file_3, format='jpg')
        ### add images of ensemble 1 for video
        if idx_ensem==0:
            images_3.append(imageio.imread(plot_file_3))
            if sim_model=='bib_ecc':
                plt.show()
        plt.close()

        ###########
        # save data
        ###########

        sim_data_file_name='sim.h5'
        sim_data_file=os.path.join(ensem_dir, sim_data_file_name)
        sim_data=h5py.File(sim_data_file,'w')
        sim_data['sld']=sld
        sim_data.close()
# save simulation video
## z = 1/4 * z_max
imageio.mimsave(os.path.join(model_param_dir,'simu_1_4.gif'), images_1, fps=2, loop=0)
## z = 2/4 * z_max
imageio.mimsave(os.path.join(model_param_dir,'simu_2_4.gif'), images_2, fps=2, loop=0)
## z = 3/4 * z_max
imageio.mimsave(os.path.join(model_param_dir,'simu_3_4.gif'), images_3, fps=2, loop=0)
