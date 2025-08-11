# Copyright (C) 2025  Helmholtz-Zentrum-Hereon

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
This plots necessary figures for joss publications

Author: Arnab Majumdar
Date: 24.06.2025
"""
import sys
import os
import warnings
import numpy as np
import h5py
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# find current dir and and ..
lib_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
# add .. in path
sys.path.append(lib_dir)
# lib imports here (if necessary)

# ignore warnings
warnings.filterwarnings("ignore")

# analytical SAS function
def J1(x):
    """
    function desc:
    spherical bessel function
    """
    # special case
    if x==0:
        return 0
    # general case
    return (np.sin(x)-x*np.cos(x))/x**2

def ball (qmax,qmin,Npts,scale,bg,sld,sld_sol,rad):
    """
    function desc:
    analytical SAS pattern of sphere
    """
    vol=(4/3)*np.pi*rad**3
    # SLD unit 10^-5 \AA^-2
    del_rho=sld-sld_sol
    q_arr=np.linspace(qmin,qmax,Npts)
    FormFactor=np.zeros(len(q_arr))
    for q_idx, q in enumerate(q_arr):
        if q==0:
            # Form factor unit 10^-5 \AA
            FormFactor[q_idx]=vol*del_rho
        else:
            # Form factor unit 10^-5 \AA
            FormFactor[q_idx]=3*vol*del_rho*J1(q*rad)/(q*rad)
    # Intensity unit 10^-10 \AA^2
    Iq_arr = (scale)*np.abs(FormFactor)**2+bg
    return Iq_arr, q_arr

def fit_func_sld_grow(q_in, sld_opt):
    """
    function desc:
    fit function
    """
    # pylint: disable=unused-variable
    Iq, q_out =ball(qmax=max(q_in),qmin=min(q_in),
                        Npts=len(q_in), scale=1 , bg=0,
                        sld=sld_opt, sld_sol=0,
                        rad=rad_grain)
    return Iq

############
# Input data
############
### file locations ###
# xml location
xml_folder='./xml/'
# script dir and working dir
script_dir = "src"  # Full path of the script
working_dir = "."  # Directory to run the script in


### struct gen ###
# box side lengths (float values)
length_a=40.
length_b=length_a
length_c=length_a
# number of cells in each direction (int values)
nx=40
ny=nx
nz=nx
# element details
el_type='lagrangian'
el_order=1
# calculate mid point of structure (simulation box)
mid_point=np.array([length_a/2, length_b/2, length_c/2])


### sim gen ###
# model name
sim_model='sld_grow'
# simulation parameters
dt=1.
t_end=10.
n_ensem=1
# calculate time array
t_arr=np.arange(0,t_end+dt, dt)

### model param ###
# model params
rad_grain=10
sld_in_0=2
sld_in_end=5
sld_out=1
qclean_sld=sld_out
# dir name for model param
model_param_dir_name = ('rad' + '_' + str(rad_grain) + '_' +
                        'sld_in_0' + '_' + str(sld_in_0) + '_' +
                        'sld_in_end' + '_' + str(sld_in_end) + '_' +
                        'sld_out' + '_' + str(sld_out)+ '_' +
                        'qclean_sld' + '_' + str(qclean_sld)
                        ).replace('.', 'p')


### scatt_cal ###
num_cat=101 # also check with 3
method_cat='extend'
sig_file='signal.h5'
scan_vec=np.array([1, 0, 0])
# these values are taken from publication
# doi: https://doi.org/10.3233/JNR-190116
Q_range=np.array([0.0029, 0.05])
num_points=100
num_orientation=10
# scatt settengs
scatt_settings='cat_' + method_cat + '_' + str(num_cat) + 'Q_' \
    + str(Q_range[0]) + '_' + str(Q_range[1]) + '_' + 'orien__' + str(num_orientation)
scatt_settings=scatt_settings.replace('.', 'p')

"""
read folder structure
"""
# folder structure
## mother folder name
### save length values as strings
### decimal points are replaced with p
length_a_str=str(length_a).replace('.','p')
length_b_str=str(length_a).replace('.','p')
length_c_str=str(length_a).replace('.','p')

### save num_cell values as strings
nx_str=str(nx)
ny_str=str(ny)
nz_str=str(nz)

### save element_order values as strings
el_order_str='lagrangian_' + str(el_order)

mother_dir_name = length_a_str+'_' + length_b_str+'_' + length_c_str\
      + '_' + nx_str + '_' + ny_str + '_' + nz_str + '_' + el_order_str
data_dir=os.path.join(working_dir, 'data')
mother_dir = os.path.join(data_dir, mother_dir_name)

# read structure info
data_file=os.path.join(mother_dir, 'structure/struct.h5')

# folder name for simulation
sim_dir=os.path.join(mother_dir, 'simulation')

# folder name for model
model_dir_name= (sim_model + '_tend_' + str(t_end) + '_dt_' + str(dt) \
    + '_ensem_' + str(n_ensem)).replace('.','p')
model_dir=os.path.join(sim_dir,model_dir_name)

# folder name for model with particular run param
model_param_dir=os.path.join(model_dir,model_param_dir_name)

# create folder for figure (one level up from data folder)
figure_dir=os.path.join(mother_dir, '../../figure/')
os.makedirs(figure_dir, exist_ok=True)
## folder for this suit of figures
plot_dir=os.path.join(figure_dir, sim_model + '_joss')
os.makedirs(plot_dir, exist_ok=True)

# initialize fit_params (radius)
sld_fit=np.zeros_like(t_arr)
sld_ana=sld_in_0+t_arr*(sld_in_end-sld_in_0)/t_end

# initialize fit_params (1: radius, 2: NA)
fig_scatt_all, ax_scatt_all = plt.subplots(figsize=(7, 5))
fig_fit_all, ax_fit_all = plt.subplots(figsize=(7, 5))

# color scheme
cmap_rainbow=cm.get_cmap('rainbow')
color_rainbow = cmap_rainbow(np.linspace(0, 1, len(t_arr)))

for i,t in enumerate(t_arr):
    t_str=str(t).replace('.','p')
    # time_dir name
    t_dir_name=f't{i:0>3}'
    t_dir=os.path.join(model_param_dir, t_dir_name)

    # box geometry
    vol_box=length_a*length_b*length_c

    # volume for normalization
    vol_norm=vol_box

    # numerical intensity
    ## read I vs Q signal file
    Iq_data_file_name=f'Iq_{scatt_settings}.h5'
    Iq_data_file=os.path.join(t_dir,Iq_data_file_name)
    Iq_data=h5py.File(Iq_data_file,'r')
    Iq_num_raw=Iq_data['Iq'][:] # unit fm^2
    Iq_num=Iq_num_raw/vol_norm*10**2 # unit (fm^2 / \AA^3) * 10^2 = cm^-1
    q_num=Iq_data['Q'][:]
    Iq_data.close()

    # fit radius w.r.t. numerical intensity
    popt, pcov = curve_fit(fit_func_sld_grow, q_num, Iq_num_raw) # pylint: disable=unbalanced-tuple-unpacking
    sld_fit[i]=popt[0]+sld_out

    # ananlytical intensity
    ## Intensity unit 10^-10 \AA^2
    Iq_ana_raw,q_ana= ball(qmax=np.max(q_num),qmin=np.min(q_num),Npts=100,
                scale=1,bg=0,sld=sld_fit[i], sld_sol=sld_out, rad=rad_grain)
    ## Normalize by volume
    ## (Before * 10**2) Intensity unit 10^-10 \AA^-1 = 10 ^-2 cm^-1
    ## (after * 10**2) Intensity unit cm^-1
    Iq_ana = (Iq_ana_raw / vol_norm) * 10**2

    # plot scatt patern for all time steps
    ax_scatt_all.loglog(q_ana, Iq_ana,
                         color=color_rainbow[i],
                           zorder=len(t_arr)+i,
                             label= f'Ana (t = {t} s)')
    ax_scatt_all.loglog(q_num, Iq_num,
                         color=color_rainbow[i],
                           zorder=i, linestyle='',
                             marker='o', markersize=3,
                               label= f'Num (t = {t} s)')

# plot formatting
## legend
# ax_scatt_all.legend(ncol=2)
## labels
ax_scatt_all.set_xlabel(r'Q [$\mathrm{\AA}^{-1}$]')
ax_scatt_all.set_ylabel(r'I(Q) [$\mathrm{cm}^{-1}$]')
## SANS upper boundary Q=1 \AA^-1
ax_scatt_all.set_xlim([Q_range[0], Q_range[1]])
# ax_scatt_all.set_ylim(bottom=1e4 )
## save plot
plot_file_name=f'Iq_{sim_model}.pdf'
plot_file=os.path.join(plot_dir,plot_file_name)
fig_scatt_all.savefig(plot_file, format='pdf')
plt.close('all')

# plot fit param for all time step
plot_file_name=f'sld_fit_{sim_model}.pdf'
plot_file=os.path.join(plot_dir,plot_file_name)
fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(t_arr, sld_ana, 'b', label= 'Simulation value')
ax.plot(t_arr, sld_fit, 'r', linestyle='', marker='^', markersize=5, label= 'Fit value')

# plot formatting
## legend
ax.legend()
## labels
ax.set_xlabel('Time [s]')
ax.set_ylabel(r'SLD of grain [$10^{-5} \cdot \mathrm{\AA}^{-2}$]')
## limits
ax.grid(True)
## save plot
plt.savefig(plot_file, format='pdf')
plt.close(fig)

sig_eff_data_file_name=f'sig_eff_{scatt_settings}.h5'
sig_eff_data_file=os.path.join(model_param_dir, sig_eff_data_file_name)
sig_eff_data=h5py.File(sig_eff_data_file,'r')
sig_eff_num=sig_eff_data['sig_eff'][:]
sig_eff_t=sig_eff_data['t'][:]
sig_eff_data.close()

contrast_arr = sld_fit-sld_out
factor=sig_eff_num[0]/(contrast_arr[0]**2)

# plot fit param for all time step
plot_file_name=f'sig_eff_fit_{sim_model}.pdf'
plot_file=os.path.join(plot_dir,plot_file_name)
fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(sig_eff_t, sig_eff_num, 'b',
         label= 'Simulation value')
ax.plot(sig_eff_t, factor*contrast_arr**2, 'r',
         linestyle='',
           marker='^', markersize=5,
             label= 'Fit value')

# plot formatting
## legend
ax.legend()
## labels
ax.set_xlabel('Time [s]')
ax.set_ylabel(r'Effective cross-section [$10^{-5} \cdot \mathrm{\AA}^{-2}$]')
## limits
ax.grid(True)
## save plot
plt.savefig(plot_file, format='pdf')
plt.close(fig)
