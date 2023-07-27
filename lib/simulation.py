
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

### 3d function ###

### static grain models ###
# spherical grain


def sph_grain_3d(coords,origin,r,sld_in,sld_out):
    coords = np.array(coords)
    sld = sld_out*np.ones(len(coords))
    cord_ed = np.sum((coords-origin)**2,axis=1)
    sld [cord_ed <= r**2] = sld_in
    return sld

### dyanmic grain models ###
# spherical grain

# diffusion into the grain
def sph_grain_diffus_3d(coords,origin,r,sld_in,sld_out,t):
    r_arr=np.linspace(r,0,len(t))
    coords = np.array(coords)
    sld = sld_out*np.ones((len(coords),len(t)))
    cord_ed = np.sum((coords-origin)**2,axis=1)
    for i in range(len(t)):
        #curr_t=t[i]
        curr_r=r_arr[i]
        sld_t = sld[:,i]
        sld_t [cord_ed <= curr_r**2] = sld_in
        sld[:,i]=sld_t
    return sld