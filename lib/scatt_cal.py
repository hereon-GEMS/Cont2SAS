
import numpy as np
from scipy import special
from scipy.optimize import minimize
#from numba import njit, prange
from scipy.special import erf
import xml.etree.ElementTree as ET
import os
import mdtraj as md


def pseudo_b(node_sld,connec,cell_vol):
    num_node=len(node_sld)
    num_cell=len(connec)
    pseudo_b=np.zeros(num_cell)
    for i in range(num_cell):
        cell_i_node_idx = connec[i,:]
        cell_i_node_sld = node_sld[cell_i_node_idx]
        cell_i_cell_sld = np.average(cell_i_node_sld)
        pseudo_b[i] = cell_vol*cell_i_cell_sld
    return pseudo_b

def pseudo_b_cat(pseudo_b,num_cat,method='extend'):
    # sorting of b_values
    b_sort=np.sort(pseudo_b)
    b_arg_sort=np.argsort(pseudo_b)
    b_min=b_sort[0]
    b_max=b_sort[-1]
    b_range=b_max-b_min
    # no categorization
    if num_cat==0:
        return pseudo_b
    else:
        if method=='simple':
            """
            range of b values
            [o......o......o      ...      o] 
              div1    div2    div3...  divN
            |o......|o......|o    ...      o|
            """
            cat_min=b_min
            cat_max=b_max
            cat_width=(cat_max-cat_min)/num_cat
            cat_val_arr=np.linspace(cat_min+cat_width/2, cat_max-cat_width/2, num_cat)
            cat_range_arr=np.linspace(cat_min,cat_max,num_cat+1)    
            b_sort_cat=np.zeros(len(b_sort))
            sort_cat=np.zeros(len(b_sort))
            for i in range(len(cat_val_arr)):
                # lower and upper bound of current category in sorted b array
                cat_low_bound=cat_range_arr[i]
                cat_low_bound_arg=len(b_sort[b_sort<cat_low_bound])
                cat_upper_bound=cat_range_arr[i+1]
                cat_upper_bound_arg=len(b_sort[b_sort<=cat_upper_bound])
                # cut sorted b array between lower and upper bound of current category
                # reset categorical b array values to b value of category
                # reset category array values to current category
                b_sort_cat[cat_low_bound_arg:cat_upper_bound_arg]=cat_val_arr[i]
                sort_cat[cat_low_bound_arg:cat_upper_bound_arg]=i#print(prop_sort_cat)
        elif method=='extend':
            """
            range of b values
                [o ......o......o      ...      o] 
              div1    div2    div3    ...     divN
            |...o...|...o...|...o...|  ...  |...o...|
            """
            cat_range_extend=(b_max-b_min)/(2*(num_cat-1))
            cat_range=b_range+2*cat_range_extend
            cat_min=b_min-cat_range_extend
            cat_max=b_max+cat_range_extend
            cat_width=(cat_max-cat_min)/num_cat
            cat_val_arr=np.linspace(cat_min+cat_width/2, cat_max-cat_width/2, num_cat)
            cat_range_arr=np.linspace(cat_min,cat_max,num_cat+1)    
            b_sort_cat=np.zeros(len(b_sort))
            sort_cat=np.zeros(len(b_sort))
            for i in range(len(cat_val_arr)):
                # lower and upper bound of current category in sorted b array
                cat_low_bound=cat_range_arr[i]
                cat_low_bound_arg=len(b_sort[b_sort<cat_low_bound])
                cat_upper_bound=cat_range_arr[i+1]
                cat_upper_bound_arg=len(b_sort[b_sort<=cat_upper_bound])
                # cut sorted b array between lower and upper bound of current category
                # reset categorical b array values to b value of category
                # reset category array values to current category
                b_sort_cat[cat_low_bound_arg:cat_upper_bound_arg]=cat_val_arr[i]
                sort_cat[cat_low_bound_arg:cat_upper_bound_arg]=i#print(prop_sort_cat)
        # undo sorting of b values
        pseudo_b[b_arg_sort]=b_sort_cat
        cat=np.zeros(len(pseudo_b), dtype='int')
        cat[b_arg_sort]=sort_cat
        return pseudo_b, cat
    
def pdb_dcd_gen(pdb_dcd_dir, pseudo_pos, pseudo_b_cat_val, pseudo_b_cat_idx):
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

def db_gen(db_dir, pseudo_b_cat_val, pseudo_b_cat_idx):
    b_val=np.unique(pseudo_b_cat_val)
    b_cat=np.unique(pseudo_b_cat_idx)
    cur_num_cat=len(b_cat)
    # copy xml files that are not reproducible by coding 
    # this can be part of further work
    # check files in database_sassena dir
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

def scattxml_gen(scatter_xml_file, signal_file,scan_vector, start_length, end_length,
                  num_points, resolution_num, sld, xlength, ylength, zlength, mid_point):
    pdb_file=os.path.join('pdb_dcd','sample.pdb')
    dcd_file=os.path.join('pdb_dcd','sample.dcd')

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
    bg= ET.SubElement(scattering, "background")
    cut= ET.SubElement(bg, "cut")
    box= ET.SubElement(cut, "box")
    sld_xml=ET.SubElement(box, "sld")
    ET.SubElement(sld_xml, "real") .text = str(sld)
    ET.SubElement(sld_xml, "imaginary") .text = str(0)
    ET.SubElement(box, "xlength").text = str(round(xlength,2))
    ET.SubElement(box, "ylength").text = str(round(ylength,2))
    ET.SubElement(box, "zlength").text = str(round(zlength,2))
    mid_pt= ET.SubElement(box, "midpoint")
    ET.SubElement(mid_pt, "x").text = str(round(mid_point[0],2))
    ET.SubElement(mid_pt, "y").text = str(round(mid_point[1],2))
    ET.SubElement(mid_pt, "z").text = str(round(mid_point[2],2))
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

def qclean_sld(model, xml_dir):
    model_xml_name='model_{0}.xml'.format(model)
    model_xml=os.path.join(xml_dir, model_xml_name)
    if model == 'gg':
        tree=ET.parse(model_xml)
        root = tree.getroot()
        return float(root.find('sld_out').text)
    elif model == 'fs':
        tree=ET.parse(model_xml)
        root = tree.getroot()
        return float(root.find('sld_out').text)
    else:
        return 0
