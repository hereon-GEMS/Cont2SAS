"""
Script for generating simulated version of detector 

Author: Arnab Majumdar
Date: 24.06.2025
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import h5py
import os

# functions

# Generate array of booleans (same size as pixels)
# 1 = no beam stop
# 0 = beam stop
def bs_mask(nx,ny,dx,dy, bs_wx, bs_wy):    
    mask=np.ones((nx,ny))
    bs_x_low=(nx*dx-bs_wx)/2
    bs_x_up=(nx*dx+bs_wx)/2
    bs_y_low=(ny*dy-bs_wy)/2
    bs_y_up=(ny*dy+bs_wy)/2
    for i in range(nx):
        cur_mask_x=mask[i,0]
        pixel_x=i*dx+dx/2
        if pixel_x<bs_x_up and pixel_x>bs_x_low:
            cur_mask_x=0
        for j in range(ny):
            cur_mask_y=mask[i,j]
            pixel_y=j*dy+dy/2
            if pixel_y<bs_y_up and pixel_y>bs_y_low:
                cur_mask_y=0
            if cur_mask_x==0 and cur_mask_y==0:
                mask[i,j]=0
    return mask

# Generate array of pixel coordinates
# (x, y, z)
def detector(nx,ny,dx,dy):
    pixel=np.zeros((nx,ny, 2))
    for i in range(nx):
        for j in range(ny):
            pixel[i,j,:]=np.array([i*dx+dx/2, j*dy+dy/2])
    #pixel=bs_checker(pixel, (nx*dx/2), (ny*dy/2))
    return pixel

"""
Input detector geometry
"""

# detector geometry details
# num pixels
nx=128
ny=128
# pixel width
dx=0.008
dy=0.008

# beam stop geometry
bs_wx=0.085
bs_wy=0.085

"""
Create simulated detector
"""

# create 2d matrices of pixel coordinates
# create mask - array of booleans (same size as pixels)
pixels_2d=detector(nx,ny,dx,dy)
mask_2d=bs_mask(nx,ny,dx,dy, bs_wx, bs_wy)

# convert 2d matrices to 1d matrices
pixels_1d=pixels_2d.reshape(nx*ny,2)
mask_1d=mask_2d.reshape(nx*ny)

# merge mask and pixel coordinates
pixels_bs_cut=pixels_1d[mask_1d==1]

# plot pixels outside beam stop
plt.scatter(pixels_bs_cut[:,0], pixels_bs_cut[:,1])
plt.axis('equal')
detector_border=patches.Rectangle((0, 0), nx*dx, nx*dx, color='k', fill=False)
beam_stop = patches.Rectangle(((nx*dx-bs_wx)/2, (nx*dy-bs_wy)/2), 0.085, 0.085,
                               color='r', fill=True)
plt.gca().add_patch(detector_border)
plt.gca().add_patch(beam_stop)
plot_file='./detector_image.jpg'
plt.savefig(plot_file, format='jpg')
plt.show()
print(f'check figure: {plot_file}')

# save detector geometry
# pixel coordinates
# pixel geometry and distribution (num, width in each dimension)
# beam stop geometry (width in each dimension)
det_geo_data_file_name='detector.h5'
det_geo_data_file=os.path.join('.',det_geo_data_file_name)
det_geo_data=h5py.File(det_geo_data_file,'w')
det_geo_data['pixel_coord']=pixels_bs_cut
det_geo_data['num_pixel_x']=nx
det_geo_data['num_pixel_y']=ny
det_geo_data['width_pixel_x']=dx
det_geo_data['width_pixel_y']=dy
det_geo_data['beam_stop_width_x']=bs_wx
det_geo_data['beam_stop_width_y']=bs_wy
det_geo_data.close()
print(f'detector gemetry saved in {det_geo_data_file}')