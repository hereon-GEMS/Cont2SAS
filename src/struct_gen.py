import numpy as np


## 3d structures ##
def struct_gen_3d (a, b, c, nx, ny, nz): 
    #a = length_a, b = length_b c = length_c
    #nx = no. of divisions in x direction (same for ny and nz)
    el=np.array([0, 1, nz+2, nz+1, (nz+1)*(ny+1), (nz+1)*(ny+1)+1, \
                 (nz+1)*(ny+1)+ny+2, (nz+1)*(ny+1)+ny+1])
    dx=a/(nx)
    dy=b/(ny)
    dz=c/(nz)
    coords=[]
    for i in range(nx+1):
        for j in range(ny+1):
            for k in range(nz+1):
                coords.append([i*dx, j*dx, k*dx])
                
    con_3d=[]
    cur=0
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                con_3d.append(list((cur)+el))
                cur=cur+1
            cur=cur+1
        cur=cur+1+nz
    return coords, con_3d

## 2d structures ##
def struct_gen_2d (a, b, nx, ny, flat=0): 
    #a = length_a, b = length_b c = length_c
    #nx = no. of divisions in x direction (same for ny and nz)
    el=np.array([0, 1, (ny+2), (ny+1)])
    dx=a/(nx)
    dy=b/(ny)
    coords=[]
    for i in range(nx+1):
        for j in range(ny+1):
            if flat ==0:
                coords.append([i*dx, j*dx])
            if flat==1:
                coords.append([i*dx, j*dx,0])
                
    con_3d=[]
    cur=0
    for i in range(nx):
        for j in range(ny):
            con_3d.append(list((cur)+el))
            cur=cur+1
        cur=cur+1
    return coords, con_3d