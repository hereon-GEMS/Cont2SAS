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
sim_dir=os.path.join('../data/',
                        length_a_str+'_'+length_b_str+'_'+length_c_str+'_'+
                        nx_str+'_'+ny_str+'_'+nz_str+'/simulation')
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
    print('calculating scattering function')
else:
    print('create simulation first')

for i in range(len(t_arr)):
    t=t_arr[i]
    # time_dir name
    t_dir_name='t{0:0>3}'.format(i)
    t_dir=os.path.join(model_param_dir, t_dir_name)
    Iq_all_ensem=np.zeros((num_points,n_ensem))
    q_all_ensem=np.zeros((num_points,n_ensem))
    for j in range(n_ensem):
        idx_ensem=j
        # create ensemble dir
        ensem_dir_name='ensem{0:0>3}'.format(idx_ensem)
        ensem_dir=os.path.join(t_dir, ensem_dir_name)
        """
        discretization
        """
        # read node sld
        sim_data_file_name='sim.h5'
        sim_data_file=os.path.join(ensem_dir, sim_data_file_name)
        sim_data=h5py.File(sim_data_file,'r')
        node_sld=sim_data['sld'][:]
        sim_data.close()

        # create scatt dir
        scatt_dir_name='scatt_cal'
        scatt_dir=os.path.join(ensem_dir, scatt_dir_name)
        # calculate pseudo atom position
        pseudo_pos=cells


        # calculate pseudo atom scattering lengths
        cell_dx=length_a/nx
        cell_dy=length_b/ny
        cell_dz=length_b/nz
        cell_vol=cell_dx*cell_dy*cell_dz
        pseudo_b=scatt.pseudo_b(node_sld,con,cell_vol)
        #sld_dyn_cell_cat, cat = procs.categorize_prop_3d_t(sld_dyn_cell, 10)

        # categorize SLD
        pseudo_b_cat_val, pseudo_b_cat_idx = scatt.pseudo_b_cat(pseudo_b,num_cat,method=method_cat)

        """
        calculate I vs Q
        """
        
        ### pdb dcd generation ###
        pdb_dcd_dir=os.path.join(scatt_dir,'pdb_dcd')
        os.makedirs(pdb_dcd_dir, exist_ok=True)
        points=np.float32(pseudo_pos)
        cat_prop=np.float32(pseudo_b_cat_val)
        topo=md.Topology() #= md.Topology()
        ch=topo.add_chain()
        res=topo.add_residue('RES', ch)
        # os.makedirs(dir_name, exist_ok=True)
        pdb_file_name=os.path.join(pdb_dcd_dir, 'sample.pdb')
        for i in range(len(points)):
            el_name='Pseudo'+str(pseudo_b_cat_idx[i])
            sym='P'+str(pseudo_b_cat_idx[i])
            try:
                ele=(md.element.Element(10,el_name, sym, 10, 10))
            except AssertionError:
                ele=(md.element.Element.getBySymbol(sym))
            topo.add_atom(sym, ele, res)
        with md.formats.PDBTrajectoryFile(pdb_file_name,'w') as f:
            f.write(points, topo) 
        dcd_file_name=os.path.join(pdb_dcd_dir, 'sample.dcd')
        with md.formats.DCDTrajectoryFile(dcd_file_name, 'w') as f:
            n_frames = 1
            for j in range(n_frames):
                f.write(points)

        ### database generator ###
        # org command
        # dsv.database_generator(min(sld_dyn_cell_cat), max(sld_dyn_cell_cat), ndiv=10, database_dir=db_dir)
        # detailed version
        db_dir_name='database'
        db_dir=os.path.join(scatt_dir,db_dir_name)
        os.makedirs(db_dir, exist_ok=True)

        b_val=np.unique(pseudo_b_cat_val)
        b_cat=np.unique(pseudo_b_cat_idx)
        cur_num_cat=len(b_cat)
        # copy xml files that are not reproducible by coding 
        # this can be part of further work
        # check files in database_sassena dir
        print('cp -r ../database_sassena/*.xml ' + db_dir + '/')
        os.system('cp -r ../database_sassena/*.xml ' + db_dir + '/')
        definition_dir=os.path.join(db_dir, 'definitions')
        os.makedirs(definition_dir, exist_ok=True)
    
        # exclusionfactors-neutron.xml
        filename='exclusionfactors-neutron.xml'
        xml_file=os.path.join(definition_dir, filename)
        
        exclusionfactors = ET.Element("exclusionfactors")
        for i in range(cur_num_cat):
            element = ET.SubElement(exclusionfactors, "element")
            el_name='Pseudo'+str(b_cat[i])
            ET.SubElement(element, "name").text = el_name
            ET.SubElement(element, "type").text = "1"
            ET.SubElement(element, "param").text = "1"   
            tree = ET.ElementTree(exclusionfactors)
        tree.write(xml_file)
    
        # masses.xml
        filename='masses.xml'
        xml_file=os.path.join(definition_dir, filename)
    
        masses = ET.Element("masses")
        for i in range(cur_num_cat):
            element = ET.SubElement(masses, "element")
            el_name='Pseudo'+str(b_cat[i])
            ET.SubElement(element, "name").text = el_name
            ET.SubElement(element, "param").text = "1"  
            tree = ET.ElementTree(masses)
        tree.write(xml_file)
        
        # names.xml
        filename='names.xml'
        xml_file=os.path.join(definition_dir, filename)
    
        names = ET.Element("names")
        pdb = ET.SubElement(names, "pdb")
        for i in range(cur_num_cat):
            element = ET.SubElement(pdb, "element")
            el_name='Pseudo'+str(b_cat[i])
            el_sym='P'+str(b_cat[i])
            ET.SubElement(element, "name").text = el_name
            ET.SubElement(element, "param").text = '^ *'+el_sym+'.*'  
            # ET.SubElement(element, "param").text = '/ '+el_sym+'/g'  
            tree = ET.ElementTree(names)
        tree.write(xml_file)

        # sizes.xml
        filename='sizes-neutron.xml'
        xml_file=os.path.join(definition_dir, filename)
    
        sizes = ET.Element("sizes")
        for i in range(cur_num_cat):
            element = ET.SubElement(sizes, "element")
            el_name='Pseudo'+str(b_cat[i])
            # el_sym='P'+str(i)
            ET.SubElement(element, "name").text = el_name
            ET.SubElement(element, "type").text = '1'
            ET.SubElement(element, "param").text = '1'  
            tree = ET.ElementTree(sizes)
        tree.write(xml_file)
    

        # scatterfactors-neutron-coherent.xml
        filename='scatterfactors-neutron-coherent.xml'
        xml_file=os.path.join(definition_dir, filename)
        
        root = ET.Element("scatterfactors")
        for i in range(cur_num_cat):
            element = ET.SubElement(root, "element")
            el_name='Pseudo'+str(b_cat[i])
            # el_sym='P'+str(i)
            ET.SubElement(element, "name").text = el_name
            ET.SubElement(element, "type").text = '0'
            ET.SubElement(element, "param").text = str(b_val[i]) 
            tree = ET.ElementTree(root)
        tree.write(xml_file)
        
        ### scatter.xml generate ###
        # original command 
        # dsv.scatterxml_generator(time_dir, sigfile='signal.h5')
        # detailed version
        scatter_xml_file_name='scatter.xml'
        scatter_xml_file=os.path.join(scatt_dir,scatter_xml_file_name)
        
        

        # filename=os.path.join(dir_name,'scatter.xml')
        # if os.path.exists(filename):
        #     shutil.rmtree(filename)
        pdb_file=os.path.join('pdb_dcd','sample.pdb')
        dcd_file=os.path.join('pdb_dcd','sample.dcd')
        # signal_file='signal.h5'
        # resolution_num=100
        # scan_vector=[1,0,0]
        # start_length=0.0
        # end_length=1.0
        # num_points=100
    
        root=ET.Element("root")
    
        #tier 1 subelements
        
        sample = ET.SubElement(root, "sample")
        database = ET.SubElement(root, "database")
        scattering = ET.SubElement(root, "scattering")
        limits = ET.SubElement(root, "limits")
    
        #sample
        
        structure = ET.SubElement(sample, "structure")
        ET.SubElement(structure, "file").text = pdb_file
        ET.SubElement(structure, "format").text = 'pdb'
        framesets = ET.SubElement(sample, "framesets")
        frameset = ET.SubElement(framesets, "frameset")
        ET.SubElement(frameset, "file").text = dcd_file
        ET.SubElement(frameset, "format").text = 'dcd'
    
        #database
        
        ET.SubElement(database, "file").text = 'database/db-neutron-coherent.xml'
    
        #scattering
        
        ET.SubElement(scattering, "type").text = 'all'
        dsp = ET.SubElement(scattering, "dsp")
        ET.SubElement(dsp, "type").text = 'square'
        signal = ET.SubElement(scattering, "signal")
        ET.SubElement(signal, "file").text = signal_file
        vectors = ET.SubElement(scattering, "vectors")
        ET.SubElement(vectors, "type").text = 'scans'
        scans= ET.SubElement(vectors, "scans")
        scan= ET.SubElement(scans, "scan")
        ET.SubElement(scan, "X").text = str(scan_vector[0])
        ET.SubElement(scan, "Y").text = str(scan_vector[1])
        ET.SubElement(scan, "Z").text = str(scan_vector[2])
        ET.SubElement(scan, "from").text = str(start_length)
        ET.SubElement(scan, "to").text = str(end_length)
        ET.SubElement(scan, "points").text = str(num_points)
        average = ET.SubElement(scattering, "average")
        orientation = ET.SubElement(average, "orientation")
        ET.SubElement(orientation, "type").text = 'vectors'
        vectors = ET.SubElement(orientation, "vectors")
        ET.SubElement(vectors, "type").text = 'sphere'
        ET.SubElement(vectors, "algorithm").text = 'boost_uniform_on_sphere'
        ET.SubElement(vectors, "resolution").text = str(resolution_num)
    
        #limits
        decomposition=ET.SubElement(limits, "decomposition")
        ET.SubElement(decomposition, "utilization").text = '0.5'
    
        tree = ET.ElementTree(root)
        tree.write(scatter_xml_file)

        # ### sassena runner ###
        parent_dir=os.getcwd()
        os.chdir(os.path.join(parent_dir,scatt_dir))
        if os.path.exists(signal_file):
            os.remove(signal_file)
        os.system('mpirun -np 8 sassena')
        os.chdir(parent_dir)

        # read and save Iq data from current ensem
        ## read
        signal_file_loc=os.path.join(scatt_dir, signal_file)
        sig_data=h5py.File(signal_file_loc,'r')
        Iq_ensem=np.sqrt(np.sum(sig_data['fq'][:]**2,axis=1))
        q_ensem=np.sqrt(np.sum(sig_data['qvectors'][:]**2,axis=1))
        sig_data.close()
        # process
        q_arg_ensem=np.argsort(q_ensem)
        Iq_ensem=Iq_ensem[q_arg_ensem]
        q_ensem=q_ensem[q_arg_ensem]
        ## save
        Iq_all_ensem[:,idx_ensem]=Iq_ensem
        q_all_ensem[:,idx_ensem]=q_ensem

    # ensem average
    Iq=np.average(Iq_all_ensem, axis=1)
    q=q_all_ensem[:,0]
    q_arg=np.argsort(q)
    Iq=Iq[q_arg]
    q=q[q_arg]
    # plotitng I vs Q in time folder
    plt.loglog(q,Iq)
    plt.xlabel('Q')
    plt.ylabel('I(Q)')
    Iq_plot_file_name='Iq.jpg'
    Iq_plot_file=os.path.join(t_dir, Iq_plot_file_name)
    plt.savefig(Iq_plot_file, format='jpg')
    plt.show()
    # saving I vs Q in time folder
    Iq_data_file_name='Iq.h5'
    Iq_data_file=os.path.join(t_dir,Iq_data_file_name)
    Iq_data=h5py.File(Iq_data_file,'w')
    Iq_data['Iq']=Iq
    Iq_data['Q']=q
    Iq_data.close()
############################ old garbage #################################
        # sim_data_file_name='sim.h5'
        # sim_data_file=os.path.join(ensem_dir, sim_data_file_name)
        # sim_data=h5py.File(sim_data_file,'r')
        # node_sld=sim_data['sld']
        # sim_data.close()
        # dsv.pdb_dcd_write(cells, sld_dyn_cell_cat, cat, pdb_dcd_dir)
        
        # ### scatter.xml generate ###
        # dsv.scatterxml_generator(time_dir, sigfile='signal.h5')
        
        # ### database generator ###
        # db_dir=os.path.join(time_dir,'database')
        # dsv.database_generator(min(sld_dyn_cell_cat), max(sld_dyn_cell_cat), ndiv=10, database_dir=db_dir)
        
        # ### sassena runner ###
        # parent_dir=os.getcwd()
        # os.chdir(os.path.join(parent_dir,time_dir))
        # os.system('mpirun -np 8 sassena')
        # os.chdir(parent_dir)
        
        # ### 
        # sigfile_name=os.path.join(time_dir,'signal.h5')
        # q_vec_t, fq0_t, fqt_t, fq_t, fq2_t = dsv.sig_read(sigfile_name)
        # neutron_count_t=0
        # for j in range(len(q_vec_t)-1):
        #     del_q=q_vec_t[j+1]-q_vec_t[j]
        #     neutron_count_t+=0.5*del_q*(fq0_t[j+1]+fq0_t[j])
        # print('neutron count is '+ str(neutron_count_t))
        # neutron_count[i]=neutron_count_t

#         """
#         run simulation
#         """
#         sld, sld_max, sld_min=sim.model_run(sim_model, nodes, mid_point, t, t_end)

#         # plottig
#         sld_3d=sld.reshape(nx+1, ny+1, nz+1)
        
#         ## z = 1/4 * z_max
#         nodes_3d=nodes.reshape(nx+1, ny+1, nz+1, 3)
#         z_val=nodes_3d[0, 0, (nz+1)//4, 2]
#         ### .T is required to exchange x and y axis 
#         ### origin is 'lower' to put it in lower left corner 
#         plt.imshow(sld_3d[:,:,(nz+1)//4].T, extent=[0, 20, 0, 20], origin='lower', vmin=sld_min, vmax=sld_max)
#         plot_file_1=os.path.join(ensem_dir,'snap_z_{}.jpg'.format(z_val))
#         plt.colorbar()
#         plt.title(' time = {0:0>3}s \n emsemble step = {1:0>3} \
#             \n z = {2}$\AA$'.format(t,idx_ensem+1,z_val))
#         plt.savefig(plot_file_1, format='jpg', bbox_inches='tight')
#         ### add images of ensemble 1 for video
#         if idx_ensem==0:  
#             images_1.append(imageio.imread(plot_file_1))
#             if sim_model=='bib_ecc':
#                 plt.show()
#         plt.close()

#         ## z = 2/4 * z_max
#         nodes_3d=nodes.reshape(nx+1, ny+1, nz+1, 3)
#         z_val=nodes_3d[0, 0, (nz+1)//2, 2]
#         ### .T is required to exchange x and y axis 
#         ### origin is 'lower' to put it in lower left corner 
#         plt.imshow(sld_3d[:,:,(nz+1)//2].T, extent=[0, 20, 0, 20], origin='lower',vmin=sld_min, vmax=sld_max)
#         plot_file_2=os.path.join(ensem_dir,'snap_z_{}.jpg'.format(z_val))
#         plt.colorbar()
#         plt.title('time = {0:0>3}s, ensmbl num = {1}, z = {2}{3}'.format(t,idx_ensem+1,z_val,r'$\mathrm{\AA}$'))
#         #plt.title(r"time = {0:0>3}s, {1}".format(t,r'$\mathrm{\AA}$'))
#         plt.savefig(plot_file_2, format='jpg')
#         ### add images of ensemble 1 for video
#         if idx_ensem==0:  
#             images_2.append(imageio.imread(plot_file_2))
#             plt.show()
#         plt.close()

#         ## z = 3/4 * z_max
#         nodes_3d=nodes.reshape(nx+1, ny+1, nz+1, 3)
#         z_val=nodes_3d[0, 0, 3*(nz+1)//4, 2]
#         ### .T is required to exchange x and y axis 
#         ### origin is 'lower' to put it in lower left corner 
#         plt.imshow(sld_3d[:,:,3*(nz+1)//4].T, extent=[0, 20, 0, 20], origin='lower',vmin=sld_min, vmax=sld_max)
#         plot_file_3=os.path.join(ensem_dir,'snap_z_{}.jpg'.format(z_val))
#         plt.colorbar()
#         plt.title(" time = {0:0>3}s, emsemble step = {1:0>3} \
#             \n z = {2}$\AA$".format(t,idx_ensem+1,z_val))
#         plt.savefig(plot_file_3, format='jpg')
#         ### add images of ensemble 1 for video
#         if idx_ensem==0:  
#             images_3.append(imageio.imread(plot_file_3))
#             if sim_model=='bib_ecc':
#                 plt.show()
#         plt.close()
#         """
#         save data
#         """
#         sim_data_file_name='sim.h5'
#         sim_data_file=os.path.join(ensem_dir, sim_data_file_name)
#         sim_data=h5py.File(sim_data_file,'w')
#         sim_data['sld']=sld
#         sim_data.close()
# # save simulation video
# ## z = 1/4 * z_max 
# imageio.mimsave(os.path.join(model_param_dir,'simu_1_4.gif'), images_1, fps=2, loop=0)
# ## z = 2/4 * z_max 
# imageio.mimsave(os.path.join(model_param_dir,'simu_2_4.gif'), images_2, fps=2, loop=0)
# ## z = 3/4 * z_max 
# imageio.mimsave(os.path.join(model_param_dir,'simu_3_4.gif'), images_3, fps=2, loop=0)