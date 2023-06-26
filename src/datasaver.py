#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Framework for saving data in hdf5 file
Created on Fri Jun 23 10:38:13 2023

@author: amajumda
"""

import h5py 
import os
import shutil
import numpy as np
import mdtraj as md
import xml.etree.cElementTree as ET

def HDFwriter(node,con, prop_node, cell, prop_cell, filename,Folder='.'):
    file_full=os.path.join(Folder,filename)
    os.makedirs(Folder, exist_ok=True)
    print(file_full)
    file=h5py.File(file_full,'w')
    file['node']=node
    file['nodeprop']=prop_node
    file['cell']=cell
    file['cellprop']=prop_cell
    file['connectivity']=con
    file.close()
    
def pdb_dcd_gen(coords, cat_prop, prop_min, prop_max, ndiv,dir_name):
    num=ndiv
    element_names=[]
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    for i in range(num):
        el_name='Pseudo'+str(i) 
        element_names.append(el_name)
    # prop_min=np.min(cat_prop)
    # prop_max=np.max(cat_prop)
    cat_mat=np.linspace(prop_min, prop_max, ndiv)
    # for i in range(len(t)):
    topo=md.Topology() #= md.Topology()
    ch=topo.add_chain()
    res=topo.add_residue('RES', ch)
    # os.makedirs(dir_name, exist_ok=True)
    pdb_file_name=os.path.join(dir_name, 'sample.pdb')
    new_coords=[]
    for j in range(len(coords)):
        # print(atoms[0][j])
        prop_val=cat_prop[0][j]
        atom_type=int(np.where(cat_mat==prop_val)[0])
        # atom_type=int(cat_prop[i][j])
        el_name=element_names[atom_type]
        sym=el_name[0]+el_name[-1]
        try:
            ele=(md.element.Element(10,el_name, sym, 10, 10))
        except AssertionError:
            ele=(md.element.Element.getBySymbol(sym))
        topo.add_atom(sym, ele, res)
    with md.formats.PDBTrajectoryFile(pdb_file_name,'w','True') as f:
        f.write(coords, topo) 
    dcd_file_name=os.path.join(dir_name, 'sample.dcd')
    with md.formats.DCDTrajectoryFile(dcd_file_name, 'w') as f:
        n_frames = 1
        for j in range(n_frames):
            f.write(coords)
            
def scatterxml_generator(dir_name, sigfile='signal.h5'):
    filename=os.path.join(dir_name,'scatter.xml')
    # if os.path.exists(filename):
    #     shutil.rmtree(filename)
    pdb_file=os.path.join('pdb_dcd','sample.pdb')
    dcd_file=os.path.join('pdb_dcd','sample.dcd')
    signal_file='signal.h5'
    resolution_num=10000
    scan_vector=[1,0,0]
    start_length=0.0
    end_length=5.0
    num_points=500
    
    root=ET.Element("root")
    
    #tier 1 subelements
    
    sample = ET.SubElement(root, "sample")
    database = ET.SubElement(root, "database")
    scattering = ET.SubElement(root, "scattering")
    
    #scattering
    
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
    ET.SubElement(signal, "file").text = sigfile
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
    # masses = ET.Element("masses")
    # for i in range(ndiv):
    #     element = ET.SubElement(masses, "element")
    #     el_name='Pseudo'+str(i)
    #     ET.SubElement(element, "name").text = el_name
    #     ET.SubElement(element, "param").text = "1"  
    tree = ET.ElementTree(root)
    tree.write(filename)
    
def database_generator(out_sl, in_sl, ndiv=10, database_dir='database'):
    sld_arr=np.linspace(out_sl, in_sl, ndiv)
        
    # database_dir='database'
    
    os.makedirs(database_dir, exist_ok=True)
    os.system('cp -r  database1/*.xml ' + database_dir + '/')
    definition_dir=os.path.join(database_dir, 'definitions')
    os.makedirs(definition_dir, exist_ok=True)
    
    """
    this part to be done later
    # # db-neutron-coherent.xml
    # filename='db-neutron-coherent.xml'
    # xml_file=os.path.join(database_dir, filename)
    # version=ET.Element("root")
    # ET.Element(version, "http://www.w3.org/2001/XInclude", name="blah")
    # tree = ET.ElementTree(version)
    # tree.write(xml_file)
    """
    
    # exclusionfactors-neutron.xml
    filename='exclusionfactors-neutron.xml'
    xml_file=os.path.join(definition_dir, filename)
    
    exclusionfactors = ET.Element("exclusionfactors")
    # element = ET.SubElement(exclusionfactors, "element")
    for i in range(ndiv):
        element = ET.SubElement(exclusionfactors, "element")
        el_name='Pseudo'+str(i)
        ET.SubElement(element, "name").text = el_name
        ET.SubElement(element, "type").text = "1"
        ET.SubElement(element, "param").text = "1"   
        tree = ET.ElementTree(exclusionfactors)
    tree.write(xml_file)
    
    # masses.xml
    filename='masses.xml'
    xml_file=os.path.join(definition_dir, filename)
    
    masses = ET.Element("masses")
    for i in range(ndiv):
        element = ET.SubElement(masses, "element")
        el_name='Pseudo'+str(i)
        ET.SubElement(element, "name").text = el_name
        ET.SubElement(element, "param").text = "5"  
        tree = ET.ElementTree(masses)
    tree.write(xml_file)
        
    # names.xml
    filename='names.xml'
    xml_file=os.path.join(definition_dir, filename)
    
    names = ET.Element("names")
    pdb = ET.SubElement(names, "pdb")
    for i in range(ndiv):
        element = ET.SubElement(pdb, "element")
        el_name='Pseudo'+str(i)
        el_sym='P'+str(i)
        ET.SubElement(element, "name").text = el_name
        ET.SubElement(element, "param").text = '^ *'+el_sym+'.*'  
        # ET.SubElement(element, "param").text = '/ '+el_sym+'/g'  
        tree = ET.ElementTree(names)
    tree.write(xml_file)
    # sizes.xml
    filename='sizes-neutron.xml'
    xml_file=os.path.join(definition_dir, filename)
    
    sizes = ET.Element("sizes")
    for i in range(ndiv):
        element = ET.SubElement(sizes, "element")
        el_name='Pseudo'+str(i)
        # el_sym='P'+str(i)
        ET.SubElement(element, "name").text = el_name
        ET.SubElement(element, "type").text = '1'
        ET.SubElement(element, "param").text = '5'  
        tree = ET.ElementTree(sizes)
    tree.write(xml_file)
    
    # # sizes.xml
    # filename='sizes.xml'
    # xml_file1=os.path.join(definition_dir, filename)
    # filename='sizes-neutron.xml'
    # xml_file2=os.path.join(definition_dir, filename)
    # os.system('cp -r  ' + xml_file1 + ' '  + xml_file2)
    
    # scatterfactors-neutron-coherent.xml
    filename='scatterfactors-neutron-coherent.xml'
    xml_file=os.path.join(definition_dir, filename)
    
    root = ET.Element("scatterfactors")
    for i in range(ndiv):
        element = ET.SubElement(root, "element")
        el_name='Pseudo'+str(i)
        # el_sym='P'+str(i)
        ET.SubElement(element, "name").text = el_name
        ET.SubElement(element, "type").text = '0'
        ET.SubElement(element, "param").text = str(sld_arr[i]) 
        tree = ET.ElementTree(root)
    tree.write(xml_file)