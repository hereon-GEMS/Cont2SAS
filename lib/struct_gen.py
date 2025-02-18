import numpy as np
import matplotlib.pyplot as plt
import h5py 

# decide the element type
def node_cell_gen_3d (a, b, c, nx, ny, nz, el_info):
    el_type = el_info['type']
    if el_type == 'lagrangian':
        el_order=el_info['order']
        return lagrangian(a, b, c, nx, ny, nz, el_order)


# generates nodes, connectivity and cells (used in the node_cell version)
def lagrangian (a, b, c, nx, ny, nz, el_order): 
    if el_order==1:
        #a = length_a, b = length_b c = length_c
        #nx = no. of divisions in x direction (same for ny and nz)
        
        # cell dimension
        dx=a/(nx)
        dy=b/(ny)
        dz=c/(nz)

        # total number of nodes and cells
        num_nodes=(nx+1)*(ny+1)*(nz+1)
        num_cells=nx*ny*nz
        
        # create node matrix
        nodes=np.zeros((num_nodes,3))
        idx=0
        for i in range(nx+1):
            for j in range(ny+1):
                for k in range(nz+1):
                    nodes[idx,:]=np.array([i*dx, j*dy, k*dz])
                    idx+=1
                    #print(idx)
        
        # create connectivity matrix
        el=np.array([0, 1, 
                     (nz+1), (nz+1)+1, 
                     (nz+1)*(ny+1),(nz+1)*(ny+1)+1, 
                     (nz+1)*(ny+1)+(nz+1), (nz+1)*(ny+1)+(nz+1)+1])
        con_3d=np.zeros((num_cells,len(el)), dtype=int)
        cur=0
        idx=0
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    con_3d[idx,:]=cur+el
                    idx+=1
                    cur=cur+1
                cur=cur+1
            cur=cur+1+nz
        
        # corner nodes for plotting mesh
        mesh=con_3d[:,[0,1,3,2,4,5,7,6]]
        
        # create cell matrix
        cells=np.zeros((num_cells,3))
        idx=0
        for connec in con_3d:
            points=nodes[connec]
            cell_center=np.average(points, axis=0)
            cells[idx]=cell_center
            idx+=1
    elif el_order==2:
        #a = length_a, b = length_b c = length_c
        #nx = no. of divisions in x direction (same for ny and nz)
        
        # cell dimension
        dx=a/(nx)
        dy=b/(ny)
        dz=c/(nz)

        # total number of nodes and cells
        num_nodes=(2*nx+1)*(2*ny+1)*(2*nz+1)
        num_cells=nx*ny*nz
        
        # create node matrix
        nodes=np.zeros((num_nodes,3))
        idx=0
        for i in range(2*nx+1):
            for j in range(2*ny+1):
                for k in range(2*nz+1):
                    nodes[idx,:]=np.array([i*dx/2, j*dy/2, k*dz/2])
                    idx+=1
        
        # create connectivity matrix
        el=np.array([0, 1, 2,
                    (2*nz+1), (2*nz+1)+1, (2*nz+1)+2,
                    2*(2*nz+1), 2*(2*nz+1)+1, 2*(2*nz+1)+2,
                    (2*nz+1)*(2*ny+1), (2*nz+1)*(2*ny+1)+1, (2*nz+1)*(2*ny+1)+2,
                    (2*nz+1)*(2*ny+1)+(2*nz+1), (2*nz+1)*(2*ny+1)+(2*nz+1)+1, (2*nz+1)*(2*ny+1)+(2*nz+1)+2,
                    (2*nz+1)*(2*ny+1)+2*(2*nz+1), (2*nz+1)*(2*ny+1)+2*(2*nz+1)+1, (2*nz+1)*(2*ny+1)+2*(2*nz+1)+2,
                    2*(2*nz+1)*(2*ny+1), 2*(2*nz+1)*(2*ny+1)+1, 2*(2*nz+1)*(2*ny+1)+2,
                    2*(2*nz+1)*(2*ny+1)+(2*nz+1), 2*(2*nz+1)*(2*ny+1)+(2*nz+1)+1, 2*(2*nz+1)*(2*ny+1)+(2*nz+1)+2,
                    2*(2*nz+1)*(2*ny+1)+2*(2*nz+1), 2*(2*nz+1)*(2*ny+1)+2*(2*nz+1)+1, 2*(2*nz+1)*(2*ny+1)+2*(2*nz+1)+2])
        con_3d=np.zeros((num_cells,len(el)), dtype=int)
        cur=0
        idx=0
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    con_3d[idx,:]=cur+el
                    idx+=1
                    cur=cur+2
                cur=cur+(2*nz+1) +1
            cur=cur+(2*nz+1) + (2*nz+1)*(2*ny+1)
        
        # corner nodes for plotting mesh
        mesh=con_3d[:,[0,2,8,6,18,20,26,24]]
        print(nodes[mesh[1,:]])
        
        # create cell matrix
        cells=np.zeros((num_cells,3))
        idx=0
        for connec in con_3d:
            points=nodes[connec]
            cell_center=np.average(points, axis=0)
            cells[idx]=cell_center
            idx+=1

    return nodes, cells, con_3d, mesh

# plot script of nodes or cells
def plotter_3d(points_3d, save_plot=False, save_dir='.',
               filename='plot', figsize=(10,10)):
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection='3d')
    #ax = plt.axes()
    points_3d=np.array(points_3d)
    ax.scatter(points_3d[:,0],points_3d[:,1],points_3d[:,2])
    if save_plot:
        plt.savefig(save_dir+'/'+filename+'.png', format='png')
    plt.close()

# plot mesh
def mesh_plotter_3d(points_3d, con_3d, save_plot=False, 
                    save_dir='.', filename='plot', figsize=(10,10)):
    points_3d=np.array(points_3d)
    fig=plt.figure(figsize=figsize)
    ax=plt.axes(projection='3d')
    for connec in con_3d:
        point=points_3d[connec]
        draw1=np.array([0,1,2,3,0])    
        ax.plot(point[draw1,0], point[draw1,1],point[draw1,2],'k')
        draw2=draw1+4
        ax.plot(point[draw2,0], point[draw2,1],point[draw2,2],'k')
        draw3=np.array([0,4])
        for i in range(4):
            draw_cur=draw3+i
            ax.plot(point[draw_cur,0], point[draw_cur,1],point[draw_cur,2],'k')
    if save_plot:
        plt.savefig(save_dir+'/'+ filename+'.png', format='png')

# write node, cell, connectivity , mesh 
def mesh_write(file_name, nodes, cells, con, mesh):
    #file_full=os.path.join(Folder,filename)
    #os.makedirs(Folder, exist_ok=True)
    file=h5py.File(file_name,'w')
    file['nodes']=nodes
    file['cells']=cells
    file['connectivity']=con
    file['mesh']=mesh
    file.close()
# read node, cell, connectivity , mesh    
def mesh_read(file_name):
    file=h5py.File(file_name,'r')
    nodes=file['nodes'][:]
    cells=file['cells'][:]
    con=file['connectivity'][:]
    mesh=file['mesh'][:]
    file.close()
    return nodes, cells, con, mesh

# function for converting string to boolean
def str_to_bool(value):
    if value.lower() in ['true', '1', 'yes']:
        return True
    elif value.lower() in ['false', '0', 'no']:
        return False
    return None







# # generates nodes, connectivity and cells (used in the node_cell version)
# def node_cell_gen_3d (a, b, c, nx, ny, nz): 
#     #a = length_a, b = length_b c = length_c
#     #nx = no. of divisions in x direction (same for ny and nz)
    
#     # cell dimension
#     dx=a/(nx)
#     dy=b/(ny)
#     dz=c/(nz)

#     # total number of nodes and cells
#     num_nodes=(nx+1)*(ny+1)*(nz+1)
#     num_cells=nx*ny*nz
    
#     # create node matrix
#     nodes=np.zeros((num_nodes,3))
#     idx=0
#     for i in range(nz+1):
#         for j in range(ny+1):
#             for k in range(nx+1):
#                 nodes[idx,:]=np.array([k*dx, j*dy, i*dz])
#                 idx+=1
#                 #print(idx)
    
#     # create connectivity matrix
#     el=np.array([0, 1, nx+2, nx+1, (nx+1)*(ny+1), (nx+1)*(ny+1)+1, \
#                   (nx+1)*(ny+1)+ny+2, (nx+1)*(ny+1)+ny+1])
#     con_3d=np.zeros((num_cells,len(el)), dtype=int)
#     cur=0
#     idx=0
#     for i in range(nz):
#         for j in range(ny):
#             for k in range(nx):
#                 con_3d[idx,:]=cur+el
#                 #con_3d.append(list((cur)+el))
#                 idx+=1
#                 cur=cur+1
#             cur=cur+1
#         cur=cur+1+nz
    
#     # create cell matrix
#     cells=np.zeros((num_cells,3))
#     idx=0
#     for connec in con_3d:
#         points=nodes[connec]
#         cell_center=np.average(points, axis=0)
#         cells[idx]=cell_center
#         idx+=1
#     return nodes, cells, con_3d

## 2d structures ##
# def struct_gen_2d (a, b, nx, ny, flat=0): 
#     #a = length_a, b = length_b c = length_c
#     #nx = no. of divisions in x direction (same for ny and nz)
#     el=np.array([0, 1, (ny+2), (ny+1)])
#     dx=a/(nx)
#     dy=b/(ny)
#     coords=[]
#     for i in range(nx+1):
#         for j in range(ny+1):
#             if flat ==0:
#                 coords.append([i*dx, j*dx])
#             if flat==1:
#                 coords.append([i*dx, j*dx,0])
                
#     con_3d=[]
#     cur=0
#     for i in range(nx):
#         for j in range(ny):
#             con_3d.append(list((cur)+el))
#             cur=cur+1
#         cur=cur+1
#     return coords, con_3d