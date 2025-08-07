# One line to give the program's name and an idea of what it does.
# Copyright (C) 2025  Arnab Majumdar

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program; if not, see
# <https://www.gnu.org/licenses/>.

"""
This reads exodus file created by moose
creates h5 file read by this program

Author: Arnab Majumdar
Date: 24.06.2025
"""
import os
import warnings
import shutil
from netCDF4 import Dataset # pylint: disable=no-name-in-module
import numpy as np
import h5py

# uncomment if plotting is needed
# # required for commented plot
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# ignore warnings
warnings.filterwarnings("ignore")

# functions
# function for sorting coordinates
# (moose to our program)
def sort_coords(x_inp,y_inp,z_inp, nx_inp, ny_inp, nz_inp):
    """
    Sort coordinate from moose to cont2sas
    """

    # first sort w.r.t. x : (x,y,z)
    x_args_inp=np.argsort(x_inp)
    sorted_x_1 = x_inp[x_args_inp]
    sorted_y_1 = y_inp[x_args_inp]
    sorted_z_1 = z_inp[x_args_inp]

    # second sort w.r.t. y for each x: (x,y,z)
    y_args_inp=np.zeros_like(x_args_inp)
    for i in range(nx_inp+1):
        y_args_inp[i*(ny_inp+1)*(nz_inp+1)
               :
               i*(ny_inp+1)*(nz_inp+1)
               +
               (ny_inp+1)*(nz_inp+1)] = (i*(ny_inp+1)*(nz_inp+1)
                                          +
                                            np.argsort(sorted_y_1[i*(ny_inp+1)*(nz_inp+1)
                                                                  :
                                                                  i*(ny_inp+1)*(nz_inp+1)
                                                                  +
                                                                  (ny_inp+1)*(nz_inp+1)]))
    sorted_x_2 = sorted_x_1[y_args_inp]
    sorted_y_2 = sorted_y_1[y_args_inp]
    sorted_z_2 = sorted_z_1[y_args_inp]

    # third sort w.r.t. z for each y, which is sorted for each x: (x,y,z)
    z_args_inp=np.zeros_like(x_args_inp)
    for i in range((nx_inp+1)*(ny_inp+1)):
        # pylint: disable=line-too-long
        z_args_inp[i*(nz_inp+1)
               :
               i*(nz_inp+1)+(nz_inp+1)]=(i*(nz_inp+1)
                                          +
                                            np.argsort(sorted_z_2[i*(nz_inp+1)
                                                                  :
                                                                  i*(nz_inp+1)
                                                                  +
                                                                  (nz_inp+1)]))
    sorted_x_3 = sorted_x_2[z_args_inp]
    sorted_y_3 = sorted_y_2[z_args_inp]
    sorted_z_3 = sorted_z_2[z_args_inp]

    return sorted_x_3, sorted_y_3, sorted_z_3, x_args_inp, y_args_inp, z_args_inp

# function for sorting variables according to coordinates
# (moose to our program)
def sort_vars(var, x_args_srt, y_args_srt, z_args_srt):
    """
    Sort variables according to sorted coordinates
    """
    return ((var[x_args_srt])[y_args_srt])[z_args_srt]

# function for calculating SLD from molar fraction
# constant rho version
def sld_cal_fe_cr_const_rho(chi_cr_inp):
    """
    calculate sld from mole fraction
    constant molar density is assumed
    """
    # params
    rho_m=0.141e-24 # moler density ([mole/cm3] * 10^-24 = [mole/AA3])
    N_avo=6.022e23
    b_fe= 9.45
    b_cr= 3.635
    sld_C2S=rho_m*N_avo*(b_cr * chi_cr_inp + b_fe * (1-chi_cr_inp))
    return sld_C2S

# both rho and molarfraction from simu
def sld_cal_fe_cr(rho_m, chi_cr_inp):
    """
    calculate sld from mole fraction
    """
    # params
    rho_m_AA=rho_m*1e-24 # moler density ([mole/cm3] * 10^-24 = [mole/AA3])
    N_avo=6.022e23
    b_fe= 9.45
    b_cr= 3.635
    sld_C2S=rho_m_AA*N_avo*(b_cr * chi_cr_inp + b_fe * (1-chi_cr_inp))
    return sld_C2S

############
# Input data
############
# file name - output from moose
exo_inp="FeCr_out.e"
# mesh details (check values in FeCr_out.i)
nx_moose=100
ny_moose=100
nz_moose=100
# time step indexes under consideration
time_idxs=[0,200,400,600,800,864]
# output file name prefix
# full name - {h5_out_name}_{time}.h5
h5_out_prefix = 'spinodal_fe_cr'
# folder name for h5 file saving
h5_dir='../moose'

#################
# output location
#################
# erase old dir and create new
shutil.rmtree(h5_dir, ignore_errors=True)
os.makedirs(h5_dir)

#################
# read moose file
#################
# initial simulation output
ds = Dataset(exo_inp, mode='r')

# uncomment below if you need to check variable names in exodus
# # print keys in exodus file
# print("Variables:")
# for var_name in ds.variables.keys():
#     last_key = list(ds.variables.keys())[-1]
#     if var_name != last_key:
#         print(var_name + ', ', end="")
#     else:
#         print(var_name + '.')

# Reading coordinates
x_moose = ds.variables['coordx'][:]
y_moose = ds.variables['coordy'][:]
z_moose = ds.variables['coordz'][:]

# sorting coordinates and coornidate arguments
x, y, z, x_args_moose, y_args_moose, z_args_moose = sort_coords(x_moose,y_moose,z_moose,
                                               nx_moose, ny_moose, nz_moose)

# loop over time
time_arr_moose = ds.variables['time_whole']
for time_idx_i, time_idx in enumerate(time_idxs):
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
    chi_cr=sort_vars(chi_cr_moose, x_args_moose, y_args_moose, z_args_moose)

    # calculate SLD from mole fraction
    sld=sld_cal_fe_cr_const_rho(chi_cr)

    # Plotting (uncomment bellow if plotting required)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # sc = ax.scatter(x, y, z, c= sld, cmap='jet')

    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.set_title(f'sld at time {time}')
    # plt.colorbar(sc, label=varname)
    # plt.show()

    # save file in hdf format
    # file name - modelname_time.h5
    # file name
    file_name=f'{h5_out_prefix}_{time:0>5}.h5'
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
