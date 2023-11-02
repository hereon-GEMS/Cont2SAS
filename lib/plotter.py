## 3d plotter functions ##

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import axes3d
import matplotlib.patches as patches
import os
import imageio

## plot xvs y

def plot_xy(x,y, filename):
    fig, ax = plt.subplots(figsize=(10,8))
    ax.plot(x, y, marker='o', ms=20, linewidth=4)
    ax.set_xlabel('t [s]', fontsize=20)
    ax.set_ylabel('r [$\AA$]', fontsize=20)
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(which='both', width=1, length=10, labelsize=20)
    ax.grid()
    plt.savefig(filename)
    plt.close()

## plot images in time folder
def img_plot_mesh(cell, nodes, nodeprop, timestep, dyn_folder, mode, sld_in, sld_out):
    # plot cell values
    nx=len(np.unique(cell[:,0]))
    ny=len(np.unique(cell[:,1]))
    nz=len(np.unique(cell[:,2]))
    length_a=np.max(nodes[:,0])
    length_b=np.max(nodes[:,1])
    length_c=np.max(nodes[:,2])
    #cell_vol=(length_a/nx)*(length_b/ny)*(length_c/nz)
    divx=length_a/nx
    divy=length_b/ny
    divz=length_c/nz
    
    # plot imshow
    sld = nodeprop[:]#[0,:]
    
    sld_shape=sld.reshape(nx+1,ny+1,nz+1)
    #print(np.max(sld_shape))
    fig, ax = plt.subplots(figsize=(10,8))
    extent=[0, nx/2, 0, ny/2]
    images = ax.imshow(sld_shape[:,:,int(nz/2+1)], cmap='viridis', 
                       vmin=min(sld_in, sld_out), vmax=max(sld_in, sld_out),
                       extent=extent, interpolation='bilinear')#, vmin=0, vmax=20)
    #print(np.max(sld_shape[:,:,21]))
    
    ax.set_title('t = {0} s, z = 10 $\AA$'.format(timestep), fontsize=20)
    ax.set_xlabel('x [$\AA$]', fontsize=20)
    ax.set_ylabel('y [$\AA$]', fontsize=20)
    cbar=fig.colorbar(images)
    cbar.ax.tick_params(labelsize=20)
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(which='both', width=1, length=10, labelsize=20)
    #plt.xticks(fontsize=30)
    #plt.yticks(fontsize=30)
    
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(0.5)
    

    for ix in range(nx):
        for jy in range(ny):
            #print(i*0.5)
            rect = patches.Rectangle((jy*divy, ix*divx), divy, divx, linewidth=0.5, edgecolor='k', facecolor='none')
            ax.add_patch(rect)
    plot_folder_t=os.path.join(dyn_folder,'t{0:0>3}/images'.format(timestep))
    os.makedirs(plot_folder_t, exist_ok=True)
    plot_file=os.path.join(plot_folder_t,'node_{1:0>3}_{0}.pdf'.format(mode,timestep))    
    plt.savefig(plot_file)
    plot_file=os.path.join(plot_folder_t,'node_{1:0>3}_{0}.png'.format(mode,timestep))    
    plt.savefig(plot_file)
    plt.close() 

## plot pseudo scatterer with mesh
def p_scatter_plot_mesh(cell, cellprop, nodes, timestep, mode, dyn_folder, sld_in, sld_out):
    #read cells
    x = cell[:,0]
    y = cell[:,1]
    z = cell[:,2]
    sld = cellprop[:]
    
    # find middle z
    num_z=len(np.unique(z))
    if num_z%2==0:
        idx=int(num_z/2)
        mid_z=np.unique(z)[idx]
        #print(mid_z)
    else:
        idx=int(np.ceil(num_z/2))
        mid_z=np.unique(z)[idx]
        #print(mid_z)
    
    x_red = x[z==mid_z]
    y_red = y[z==mid_z]
    #z_red = z[z==mid_z]
    sld_red=sld[z==mid_z]
    
    # plot cell values
    nx=len(np.unique(x))
    ny=len(np.unique(y))
    nz=len(np.unique(z))
    length_a=np.max(nodes[:,0])
    length_b=np.max(nodes[:,1])
    length_c=np.max(nodes[:,2])
    cell_vol=(length_a/nx)*(length_b/ny)*(length_c/nz)
    divx=length_a/nx
    divy=length_b/ny
    #divz=length_c/nz
    #area=np.ones(nx*ny)
    fig, ax = plt.subplots(figsize=(10,8)) 

    scatt_plot = ax.scatter(x_red, y_red, s=40, c=sld_red, edgecolors=None,
                            cmap='viridis', vmin=min(sld_in, sld_out)*cell_vol,
                            vmax=max(sld_in, sld_out)*cell_vol)

    ax.set_title('t = {0} s, z = 10.25 $\AA$'.format(timestep), fontsize=20)
    ax.set_xlabel('x [$\AA$]', fontsize=20)
    ax.set_ylabel('y [$\AA$]', fontsize=20)
    #cbar=fig.colorbar(images)
    #cbar.ax.tick_params(labelsize=20)
    #ax.set_aspect('equal', adjustable='box')
    ax.tick_params(which='both', width=1, length=10, labelsize=20)
    ax.set_xlim([0,20])
    ax.set_ylim([0,20])
    cbar=fig.colorbar(scatt_plot)
    cbar.ax.tick_params(labelsize=20)
    #ax.set_title('t = {0}'.format(i))
    ax.set_aspect('equal', adjustable='box')

    for ix in range(nx):
        for jy in range(ny):
            #print(i*0.5)
            rect = patches.Rectangle((jy*divy, ix*divx), divy, divx, linewidth=0.5, edgecolor='k', facecolor='none')
            ax.add_patch(rect)
    plot_folder_t=os.path.join(dyn_folder,'t{0:0>3}/images'.format(timestep))
    os.makedirs(plot_folder_t, exist_ok=True)
    plot_file=os.path.join(plot_folder_t,'cat_cell_{1:0>3}_{0}.pdf'.format(mode,timestep))
    plt.savefig(plot_file, bbox_inches='tight')
    plot_file=os.path.join(plot_folder_t,'cat_cell_{1:0>3}_{0}.png'.format(mode,timestep))
    plt.savefig(plot_file, bbox_inches='tight')
    plt.close()

def plotter_3d(points_3d, save_plot=False, save_dir='.',
               filename='plot', figsize=(10,10)):
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection='3d')
    #ax = plt.axes()
    points_3d=np.array(points_3d)
    ax.scatter(points_3d[:,0],points_3d[:,1],points_3d[:,2])
    if save_plot:
        plt.savefig(save_dir+'/'+filename, format='png')
    plt.close() 
    
def mesh_plotter_3d(points_3d, con_3d, save_plot=False, 
                    save_dir='.', filename='plot', figsize=(10,10)):
    points_3d=np.array(points_3d)
    fig=plt.figure(figsize=figsize)
    ax=plt.axes(projection='3d')
    for connec in con_3d:
        #print(connec)
        point=points_3d[connec]
        draw1=np.array([0,1,2,3,0])
        #points=np.concatenate((points[0:4], points[4:8,:]),axis=0)        
        ax.plot(point[draw1,0], point[draw1,1],point[draw1,2],'k')
        draw2=draw1+4
        ax.plot(point[draw2,0], point[draw2,1],point[draw2,2],'k')
        draw3=np.array([0,4])
        for i in range(4):
            draw_cur=draw3+i
            ax.plot(point[draw_cur,0], point[draw_cur,1],point[draw_cur,2],'k')
    if save_plot:
        plt.savefig(save_dir+'/'+ filename, format='png')
        
# def colorplot_node_3d(coords, sld, nx, ny, nz, save_plot=False,save_dir='.'):
#     fig = plt.figure()
#     sld_reshape=np.array(sld).reshape(nx+1, ny+1, nz+1)
#     print(sld_reshape.shape)
#     x=np.array(coords)[:,0].reshape(nx+1, ny+1,nz+1)
#     y=np.array(coords)[:,1].reshape(nx+1, ny+1, nz+1)
#     z=np.array(coords)[:,2].reshape(nx+1, ny+1, nz+1)
#     print(int((nz+1)/2))
#     plt.title('z = ' + str(z[0,0,int((nz+1)/2)]))
#     plt.imshow(sld_reshape[:,:,int((nz+1)/2)],extent=(np.min(x), np.max(x), np.min(y), np.max(y)))
#     plt.colorbar()
#     if save_plot:
#         plt.savefig(save_dir+'/sld_node', format='png')
        
def colorplot_node_3d(nodes, sld, nx, ny, nz, save_plot=False,save_dir='.'):
    fig = plt.figure()
    sld_reshape=np.array(sld).reshape(nx+1, ny+1, nz+1)
    x=np.array(nodes)[:,0].reshape(nx+1, ny+1,nz+1)
    y=np.array(nodes)[:,1].reshape(nx+1, ny+1, nz+1)
    z=np.array(nodes)[:,2].reshape(nx+1, ny+1, nz+1)
    plt.title('z = ' + str(z[0,0,int((nz+1)/2)]))
    plt.imshow(sld_reshape[:,:,int((nz+1)/2)],extent=(np.min(x), np.max(x), np.min(y), np.max(y)))
    plt.colorbar()
    if save_plot:
        plt.savefig(save_dir+'/sld_node', format='png')
        
# def mesh_plotter_3d(coords_3d, con_3d, save_plot=False,save_dir='.'):
#     coords_3d=np.array(coords_3d)
#     fig=plt.figure()
#     ax=plt.axes(projection='3d')
#     for connec in con_3d:
#         #print(connec)
#         points=coords_3d[connec]
#         draw1=np.array([0,1,2,3,0])
#         #points=np.concatenate((points[0:4], points[4:8,:]),axis=0)        
#         ax.plot(points[draw1,0], points[draw1,1],points[draw1,2],'k')
#         draw2=draw1+4
#         ax.plot(points[draw2,0], points[draw2,1],points[draw2,2],'k')
#         draw3=np.array([0,4])
#         for i in range(4):
#             draw_cur=draw3+i
#             ax.plot(points[draw_cur,0], points[draw_cur,1],points[draw_cur,2],'k')
#     if save_plot:
#         plt.savefig(save_dir+'/mesh', format='png')
    
def colorplot_cell_3d(coords, sld, nx, ny, nz, save_plot=False,save_dir='.'):
    fig = plt.figure()
    ax = plt.axes()
    sld_reshape=np.array(sld).reshape(nx, ny, nz)
    #print(sld_reshape.shape)
    x=np.array(coords)[:,0].reshape(nx, ny, nz)
    #print(x.shape)
    y=np.array(coords)[:,1].reshape(nx, ny, nz)
    z=np.array(coords)[:,2].reshape(nx, ny, nz)
    plt.title('z = ' + str(z[0,0,int((nz+1)/2)]))
    plt.imshow(sld_reshape[:,:,int((nz+1)/2)],extent=(np.min(x), np.max(x), np.min(y), np.max(y)))
    plt.colorbar()
    if save_plot:
        plt.savefig(save_dir+'/sld_cell', format='png')


"""
def mesh_plotter_3d(coords_3d, con_3d):
    coords_3d=np.array(coords_3d)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    for connec in con_3d:
        points=coords_3d[connec]
        rect1=np.concatenate((points[0:4,:], points[0:1,:]),axis=0)
        ax.plot3D(rect1[:,0], rect1[:,1], rect1[:,2], 'k')
        rect2=np.concatenate((points[4:,:], points[4:5,:]),axis=0)
        ax.plot3D(rect2[:,0], rect2[:,1], rect2[:,2], 'k')
        # points=np.concatenate((points, points[0:1,:]),axis=0)
        for i in range(4):
            line=np.concatenate((points[i:i+1,:], points[i+4:i+5,:]),axis=0)
            ax.plot3D(line[:,0], line[:,1], line[:,2], 'k')
"""
"""
def colorplot_cell_3d(coords_3d, prop_3d_cell_t, nx, ny, nz, slice_id='z'):
    prop_3d_cell_t_reshape=np.array(prop_3d_cell_t).reshape(nx, ny, nz)
    x=np.array(coords_3d)[:,0].reshape(nx+1, ny+1, nz+1)
    y=np.array(coords_3d)[:,1].reshape(nx+1, ny+1, nz+1)
    z=np.array(coords_3d)[:,2].reshape(nx+1, ny+1, nz+1)
    if slice_id=='x':
        idx=nx//2
        x_plot=y[idx, :, :]
        y_plot=z[idx, :, :]
        prop_3d_cell_t_plot=prop_3d_cell_t_reshape[idx, :, :]
    elif slice_id=='y':
        idx=ny//2
        x_plot=x[:, idx, :]
        y_plot=z[:, idx, :]
        prop_3d_cell_t_plot=prop_3d_cell_t_reshape[:, idx, :]
    elif slice_id=='z':
        idx=nz//2
        x_plot=x[:, :, idx]
        y_plot=y[:, :, idx]
        prop_3d_cell_t_plot=prop_3d_cell_t_reshape[:, :, idx]
    # fig, ax = plt.subplots()
    plt.pcolor(x_plot, y_plot, prop_3d_cell_t_plot)
    plt.axis('equal')
    plt.axis('square')
    # ax.set_aspect('equal', 'box')
    # plt.xlim(np.min(x_plot), np.max(x_plot))
    # plt.show()

def colorplot_node_3d(coords_3d, prop_3d_node_t, nx, ny, nz, slice_id='z'):
    prop_3d_node_t_reshape=np.array(prop_3d_node_t).reshape(nx+1, ny+1, nz+1)
    x=np.array(coords_3d)[:,0].reshape(nx+1, ny+1, nz+1)
    y=np.array(coords_3d)[:,1].reshape(nx+1, ny+1, nz+1)
    z=np.array(coords_3d)[:,2].reshape(nx+1, ny+1, nz+1)
    if slice_id=='x':
        idx=nx//2
        x_plot=y[idx, :, :]
        y_plot=z[idx, :, :]
        prop_3d_node_t_plot=prop_3d_node_t_reshape[idx, :, :]
    elif slice_id=='y':
        idx=ny//2
        x_plot=x[:, idx, :]
        y_plot=z[:, idx, :]
        prop_3d_node_t_plot=prop_3d_node_t_reshape[:, idx, :]
    elif slice_id=='z':
        idx=nz//2
        x_plot=x[:, :, idx]
        y_plot=y[:, :, idx]
        prop_3d_node_t_plot=prop_3d_node_t_reshape[:, :, idx]
    plt.imshow(prop_3d_node_t_plot,extent=(np.min(x_plot), np.max(x_plot), \
                                      np.min(y_plot), np.max(y_plot)))
"""
# plot 2d strcutures #

def plotter_2d(coords_2d,show_plot=False):
    fig = plt.figure()
    #ax = plt.axes(projection='3d')
    ax = plt.axes()
    coords_2d=np.array(coords_2d)
    ax.scatter(coords_2d[:,0],coords_2d[:,1])
    if show_plot==True:
        plt.show()
    plt.close()

def colorplot_node_2d(coords, sld, nx, ny,show_plot=False):
    fig = plt.figure()
    sld_reshape=np.array(sld).reshape(nx+1, ny+1)
    print(sld_reshape.shape)
    x=np.array(coords)[:,0].reshape(nx+1, ny+1)
    y=np.array(coords)[:,1].reshape(nx+1, ny+1)
    plt.imshow(sld_reshape,extent=(np.min(x), np.max(x), np.min(y), np.max(y)))
    plt.colorbar()
    if show_plot:
        plt.show()

def mesh_plotter_2d(coords_2d, con):
    coords_2d=np.array(coords_2d)
    fig=plt.figure()
    ax=plt.axes()
    for connec in con:
        points=coords_2d[connec]
        points=np.concatenate((points, points[0:1,:]),axis=0)        
        ax.plot(points[:,0], points[:,1],'k')
    #plt.show()

def colorplot_cell_2d(coords, sld, nx, ny,show_plot=False):
    fig = plt.figure()
    ax = plt.axes()
    sld_reshape=np.array(sld).reshape(nx, ny)
    print(sld_reshape.shape)
    x=np.array(coords)[:,0].reshape(nx, ny)
    print(x.shape)
    y=np.array(coords)[:,1].reshape(nx, ny)
    plt.imshow(sld_reshape,extent=(np.min(x), np.max(x), np.min(y), np.max(y)))
    plt.colorbar()
    if show_plot:
        plt.show()
