
import numpy as np

### 2d function ###

### static grain models ###
# spherical grain

def sph_grain(coords,origin,r,sld_in,sld_out):
    coords = np.array(coords)
    sld = sld_out*np.ones(len(coords))
    cord_ed = np.sum((coords-origin)**2,axis=1)
    sld [cord_ed <= r**2] = sld_in
    return sld

### 2d function ###

### static grain models ###
# spherical grain


def sph_grain_3d(coords,origin,r,sld_in,sld_out):
    coords = np.array(coords)
    sld = sld_out*np.ones(len(coords))
    cord_ed = np.sum((coords-origin)**2,axis=1)
    sld [cord_ed <= r**2] = sld_in
    return sld