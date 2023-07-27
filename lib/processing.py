import numpy as np
import h5py

# common

def hdfreader(file):
    file_read=h5py.File(file, 'r')
    node=np.array(file_read['node'])
    cell=np.array(file_read['cell'])
    sld_cell=np.array(file_read['cellprop'])
    sld_node=np.array(file_read['nodeprop'])
    con=np.array(file_read['connectivity'])
    file_read.close()
    return node, cell, sld_cell, sld_node, con

def signalreader(sig_file):
    file_read=h5py.File(sig_file, 'r')
    q=np.sqrt(np.sum(np.array(file_read['qvectors'])**2,axis=1))
    q_arg=np.argsort(q)
    q=q[q_arg]
    fq0=np.sqrt(np.sum(np.array(file_read['fq0'])**2,axis=1))
    fq0=fq0[q_arg]
    return q, fq0

def categorize_prop(prop,t,ndiv):
    cat_prop= []
    cat_idx=[]
    for i in range(len(t)):
        prop_t=prop[i]
        prop_t_min=np.min(prop_t)
        prop_t_max=np.max(prop_t)
        step = (prop_t_max - prop_t_min)/ndiv
        cat_arr=np.linspace(prop_t_min-step/2, prop_t_max+step/2, ndiv+1)
        # step = (in_arr_max - in_arr_min)/ndiv
        cat_idx_t=[]
        # cat_prop=np.zeros(len(prop_t[i]))
        for j in range(len(cat_arr)-1):
            index_low=np.where(prop_t>=(cat_arr[j]))
            index_high=np.where(prop_t<(cat_arr[j+1]))
            # idx
            cat_idx_t.append(np.intersect1d(index_high,index_low))
        cat_idx.append(cat_idx_t)
        cat_prop_t=np.zeros(len(prop_t))
        cat_vals=np.linspace(prop_t_min,prop_t_max,ndiv)
        for j in range(ndiv):
            cat_prop_t[cat_idx_t[j]]=cat_vals[j]
        cat_prop.append(cat_prop_t)
    return cat_prop, cat_idx

def categorize_prop_dyn(prop,t,ndiv):
    cat_prop= []
    cat_idx=[]
    for i in range(len(t)):
        prop_t=prop[:,i]
        prop_t_min=np.min(prop_t)
        prop_t_max=np.max(prop_t)
        step = (prop_t_max - prop_t_min)/ndiv
        cat_arr=np.linspace(prop_t_min-step/2, prop_t_max+step/2, ndiv+1)
        # step = (in_arr_max - in_arr_min)/ndiv
        cat_idx_t=[]
        # cat_prop=np.zeros(len(prop_t[i]))
        for j in range(len(cat_arr)-1):
            index_low=np.where(prop_t>=(cat_arr[j]))
            index_high=np.where(prop_t<(cat_arr[j+1]))
            # idx
            cat_idx_t.append(np.intersect1d(index_high,index_low))
        cat_idx.append(cat_idx_t)
        cat_prop_t=np.zeros(len(prop_t))
        cat_vals=np.linspace(prop_t_min,prop_t_max,ndiv)
        for j in range(ndiv):
            cat_prop_t[cat_idx_t[j]]=cat_vals[j]
        cat_prop.append(cat_prop_t)
    return cat_prop, cat_idx

#3d structures
    

#old version with append
"""
def node2cell_3d(nodes,connec,prop, t_arr):   
    cells=[]
    cell_props=[]
    for t in range(len(t_arr)):
        cells_t=[]
        cell_props_t=[]
        for i in range(len(connec)):
            cells_t.append(np.average(np.array(nodes)[connec[i]],axis=0))
            cell_props_t.append(np.average(np.array(prop[t])[connec[i]],axis=0))
        cells.append(cells_t)
        cell_props.append(cell_props_t)
    return np.array(cells[0]), np.array(cell_props)

"""

def node2cell_3d(nodes,connec,prop, t_arr, nx, ny, nz):
    steps=len(t_arr)
    tot_nodes=(nx+1)*(ny+1)*(nz+1)
    tot_cells=(nx)*(ny)*(nz)
    cells=np.zeros((tot_cells,3,steps))
    cell_props=np.zeros((steps,tot_cells))
    for t in range(steps):
        cells_t=np.zeros((tot_cells,3))
        cell_props_t=np.zeros((tot_cells))
        for i in range(len(connec)):
            cells_t[i,:]=np.average(np.array(nodes)[connec[i]],axis=0)
            cell_props_t[i]=np.average(np.array(prop[t])[connec[i]],axis=0)
        cells[:,:,t]=cells_t
        cell_props[t,:]=cell_props_t
    return cells[:,:,0], cell_props




#2d strctures
"""
def node2Cell_2d(nodes, connec, prop, t_arr):
    cells=[]
    cell_props=[]
    for i in range(len(t_arr)):    
        for j in range(len(connec)):
            cell_nodes=connec[j,:]
            
        cells.append(np.average(np.array(nodes)[connec[i]],axis=0))
        cell_prop=np.array(prop[0])[connec[i]]
        cell_props.append(np.median(np.array(prop[9])[connec[i]],axis=0))
    return cells, cell_props
"""
def node2cell(nodes,connec,prop, t_arr):
    cells=[]
    cell_props=[]
    for t in range(len(t_arr)):
        cells_t=[]
        cell_props_t=[]
        for i in range(len(connec)):
            cells_t.append(np.average(np.array(nodes)[connec[i]],axis=0))
            cell_props_t.append(np.average(np.array(prop[t])[connec[i]],axis=0))
        cells.append(cells_t)
        cell_props.append(cell_props_t)
    return np.array(cells[0]), np.array(cell_props)



  