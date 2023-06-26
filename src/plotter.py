## 3d plotter functions ##

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import axes3d

## plot 3d structures


def plotter_3d(coords_3d,show_plot=False):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    #ax = plt.axes()
    coords_3d=np.array(coords_3d)
    ax.scatter(coords_3d[:,0],coords_3d[:,1],coords_3d[:,2])
    if show_plot:
        plt.show()
        
def colorplot_node_3d(coords, sld, nx, ny, nz, show_plot=False):
    fig = plt.figure()
    sld_reshape=np.array(sld).reshape(nx+1, ny+1, nz+1)
    print(sld_reshape.shape)
    x=np.array(coords)[:,0].reshape(nx+1, ny+1,nz+1)
    y=np.array(coords)[:,1].reshape(nx+1, ny+1, nz+1)
    z=np.array(coords)[:,2].reshape(nx+1, ny+1, nz+1)
    print(int((nz+1)/2))
    plt.title('z = ' + str(z[0,0,int((nz+1)/2)]))
    plt.imshow(sld_reshape[:,:,int((nz+1)/2)],extent=(np.min(x), np.max(x), np.min(y), np.max(y)))
    plt.colorbar()
    if show_plot:
        plt.show()
        
def mesh_plotter_3d(coords_3d, con_3d):
    coords_3d=np.array(coords_3d)
    fig=plt.figure()
    ax=plt.axes(projection='3d')
    for connec in con_3d:
        points=coords_3d[connec]
        draw1=np.array([0,1,2,3,0])
        #points=np.concatenate((points[0:4], points[4:8,:]),axis=0)        
        ax.plot(points[draw1,0], points[draw1,1],points[draw1,2],'k')
        draw2=draw1+4
        ax.plot(points[draw2,0], points[draw2,1],points[draw2,2],'k')
        draw3=np.array([0,4])
        for i in range(4):
            draw_cur=draw3+i
            ax.plot(points[draw_cur,0], points[draw_cur,1],points[draw_cur,2],'k')
    #plt.show()
    
def colorplot_cell_3d(coords, sld, nx, ny, nz,show_plot=False):
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
    if show_plot:
        plt.show()


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
