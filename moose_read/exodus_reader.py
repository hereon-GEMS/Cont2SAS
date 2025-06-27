"""
This reads exodus file created by moose
creates h5 file read by this program

Author: Arnab Majumdar
Date: 24.06.2025
"""

from netCDF4 import Dataset
import numpy as np
import h5py
import os
import shutil

# required for commented plot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

# functions
# function for sorting coordinates 
# (moose to our program)
def sort_coords(x_inp,y_inp,z_inp):

    # first sort w.r.t. x : (x,y,z)
    x_args=np.argsort(x_inp)
    sorted_x_1 = x_inp[x_args]
    sorted_y_1 = y_inp[x_args]
    sorted_z_1 = z_inp[x_args]

    # second sort w.r.t. y for each x: (x,y,z)
    y_args=np.zeros_like(x_args)
    for i in range(101):
        y_args[i*101*101:i*101*101+101*101]=i*101*101+np.argsort(sorted_y_1[i*101*101:i*101*101+101*101])
    sorted_x_2 = sorted_x_1[y_args]
    sorted_y_2 = sorted_y_1[y_args]
    sorted_z_2 = sorted_z_1[y_args]

    # third sort w.r.t. z for each y, which is sorted for each x: (x,y,z)
    z_args=np.zeros_like(x_args)
    for i in range(101*101):
        z_args[i*101:i*101+101]=i*101+np.argsort(sorted_z_2[i*101:i*101+101])
    sorted_x_3 = sorted_x_2[z_args]
    sorted_y_3 = sorted_y_2[z_args]
    sorted_z_3 = sorted_z_2[z_args]

    return sorted_x_3, sorted_y_3, sorted_z_3, x_args, y_args, z_args

# function for sorting variables according to coordinates 
# (moose to our program)
def sort_vars(var, x_args, y_args, z_args):
    return ((var[x_args])[y_args])[z_args]

# function for calculating SLD from molar fraction
# constant rho version
def sld_cal_fe_cr_const_rho(chi_cr):
    # params
    rho_m=0.141e-24 # moler density ([mole/cm3] * 10^-24 = [mole/AA3])
    N_avo=6.022e23
    b_fe= 9.45
    b_cr= 3.635 
    sld=rho_m*N_avo*(b_cr * chi_cr + b_fe * (1-chi_cr))
    return sld

# both rho and molarfraction from simu
def sld_cal_fe_cr(rho_m, chi_cr):
    # params
    rho_m_AA=rho_m*1e-24 # moler density ([mole/cm3] * 10^-24 = [mole/AA3])
    N_avo=6.022e23
    b_fe= 9.45
    b_cr= 3.635 
    sld=rho_m_AA*N_avo*(b_cr * chi_cr + b_fe * (1-chi_cr))
    return sld

"""
Input data
"""
# file name - output from moose
exo_inp="FeCr_out.e"
# time step indexes under consideration
time_idxs=[0,200,400,600,800,864]
# output file name prefix
# full name - {h5_out_name}_{time}.h5
h5_out_prefix = 'spinodal_fe_cr'
# folder name for h5 file saving 
h5_dir='../moose'

"""
output location
"""
# erase old dir and create new
shutil.rmtree(h5_dir, ignore_errors=True)
os.makedirs(h5_dir)

"""
read moose file
"""
# initial simulation output
ds = Dataset(exo_inp, mode='r')

# uncomment below if you need to check variable names in exodus
"""
# print keys in exodus file
print("Variables:")
for var_name in ds.variables.keys():
    last_key = list(ds.variables.keys())[-1]
    if var_name != last_key:
        print(var_name + ', ', end="")
    else:
        print(var_name + '.')
"""

# Reading coordinates
x_moose = ds.variables['coordx'][:]
y_moose = ds.variables['coordy'][:]
z_moose = ds.variables['coordz'][:]

# sorting coordinates and coornidate arguments
x, y, z, x_args, y_args, z_args = sort_coords(x_moose,y_moose,z_moose)

# loop over time
time_arr_moose = ds.variables['time_whole']
for i, time_idx in enumerate(time_idxs):
    # read time
    time=time_arr_moose[time_idx]
    # remove .0 from int values of time
    time=int(time) if (time%1==0) else time


    # read mole fraction
    # mole fraction is saved in 1st variable
    varname = 'vals_nod_var1'
    # read for chosen time steps
    chi_cr_moose = ds.variables[varname][time_idx, :] # [time, node]  
    # sort mole fraction
    chi_cr=sort_vars(chi_cr_moose, x_args, y_args, z_args)#((chi_cr_moose[x_args])[y_args])[z_args] 

    # calculate SLD from mole fraction
    sld=sld_cal_fe_cr_const_rho(chi_cr)

    # Plotting (uncomment bellow if plotting required)
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(x, y, z, c= sld, cmap='jet')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'sld at time {time}')
    plt.colorbar(sc, label=varname)
    plt.show()
    """

    # save file in hdf format
    # file name - modelname_time.h5
    # file name
    file_name='{0}_{1:0>5}.h5'.format(h5_out_prefix, time)
    file_loc=os.path.join(h5_dir, file_name)
    file=h5py.File(file_loc,'w')
    file['SLD']=sld # SLD obtained from simu
    file['time']=time # time step
    # params for struct_gen
    file['nx']=int(len(x)**(1/3))
    file['ny']=int(len(y)**(1/3))
    file['nz']=int(len(z)**(1/3))
    file['length_a']=int(max(x))*10 # (*10 nm to AA)
    file['length_b']=int(max(y))*10
    file['length_c']=int(max(z))*10
    # qclean sld (strategy - average of all sld)
    file['qclean_sld']=np.average(sld)
    file.close()

# close exodus file
ds.close()
