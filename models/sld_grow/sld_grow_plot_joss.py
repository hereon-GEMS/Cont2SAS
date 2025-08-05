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
This plots necessary figures for chemical composition change model
Plots are saved in figure folder

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
from matplotlib.patches import Rectangle
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
    # pylint: disable=too-many-arguments
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
    for j in range(n_ensem):
        idx_ensem=j
        # create ensemble dir
        ensem_dir_name=f'ensem{idx_ensem:0>3}'
        ensem_dir=os.path.join(t_dir, ensem_dir_name)

        # create scatt dir
        scatt_dir_name='scatt_cal_'+ scatt_settings
        scatt_dir=os.path.join(ensem_dir, scatt_dir_name)


        """
        read pseudo atom info from scatt_cal.h5
        """
        # read node sld
        # save pseudo atom info
        scatt_cal_dir_name='scatt_cal_' + scatt_settings
        scatt_cal_dir=os.path.join(ensem_dir, scatt_cal_dir_name)
        scatt_cal_data_file_name='scatt_cal.h5'
        scatt_cal_data_file=os.path.join(scatt_cal_dir, scatt_cal_data_file_name)
        scatt_cal_data=h5py.File(scatt_cal_data_file,'r')
        node_pos=scatt_cal_data['node_pos'][:]
        node_pos=np.array(node_pos)
        node_sld=scatt_cal_data['node_sld'][:]
        node_sld=np.array(node_sld)
        pseudo_pos=scatt_cal_data['pseudo_pos'][:]
        pseudo_pos=np.array(pseudo_pos)
        pseudo_b=scatt_cal_data['pseudo_b'][:]
        pseudo_b=np.array(pseudo_b)
        pseudo_b_cat_val=scatt_cal_data['pseudo_b_cat_val'][:]
        pseudo_b_cat_val=np.array(pseudo_b_cat_val)
        pseudo_b_cat_idx=scatt_cal_data['pseudo_b_cat_idx'][:]
        scatt_cal_data.close()

        # determine sld min and max for plotting
        sld_min=np.min([sld_in_0, sld_in_end, sld_out, 0])
        sld_max=np.max([sld_in_0, sld_in_end, sld_out, 0])


        # if idx_ensem==0:
        #     # plotting node SLD
        #     ## cutting at z = cut_frac * length_z
        #     cut_frac=0.5
        #     node_pos_3d=node_pos.reshape((nx+1, ny+1, nz+1, 3))
        #     z_idx= np.floor(cut_frac*(nz+1)).astype(int)
        #     z_val=node_pos_3d[0, 0, z_idx , 2]
        #     ## figure specification
        #     plot_file_name=f'SLD_{sim_model}_{t_str}.pdf'
        #     plot_file=os.path.join(plot_dir,plot_file_name)
        #     fig, ax = plt.subplots(figsize=(5, 5))
        #     ## image plot
        #     ### .T is required to exchange x and y axis
        #     ### origin is 'lower' to put it in lower left corner
        #     node_sld_3d=node_sld.reshape((nx+1, ny+1, nz+1))
        #     img = ax.imshow(node_sld_3d[:,:,z_idx].T,
        #                      extent=[0, length_a, 0, length_b],
        #                        origin='lower', vmin=sld_min, vmax=sld_max, interpolation='bilinear')
        #     ## color bar
        #     cbar = plt.colorbar(img, ax=ax)  # Add colorbar to subplot 1
        #     cbar_label=r"Scattering length density (SLD) [$10^{-5} \cdot \mathrm{\AA}^{-2}$]"
        #     cbar.set_label(cbar_label, labelpad=10)
        #     ## plot title
        #     title_text=" Cut at Z = {0} {1}".format(z_val, r"$\mathrm{\AA}$")
        #     ax.set_title(title_text)
        #     # labels
        #     ax.set_xlabel(r'X [$\mathrm{\AA}$]')
        #     ax.set_ylabel(r'Y [$\mathrm{\AA}$]')
        #     ## add mesh
        #     cell_x=length_a/nx
        #     cell_y=length_a/ny
        #     for idx1 in range(nx):
        #         for idx2 in range(ny):
        #             rect_center_x=idx1*cell_x
        #             rect_center_y=idx2*cell_y
        #             ax.add_patch(Rectangle((rect_center_x, rect_center_y), cell_x, cell_y,
        #                                     edgecolor='k', facecolor='none', linewidth=0.5))
        #     plt.savefig(plot_file, format='pdf')
        #     plt.close(fig)

        #     # plotting pseudo atoms
        #     ## cutting at z = z_val + length_z
        #     # cell_z= length_c/nz
        #     # z_val_pseudo=z_val+cell_z
        #     cut_frac=0.5
        #     pseudo_pos_3d=pseudo_pos.reshape((nx, ny, nz, 3))
        #     pseudo_b_3d=pseudo_b.reshape((nx, ny, nz))
        #     z_idx_pseudo= z_idx-1
        #     z_val_pseudo=pseudo_pos_3d[0, 0, z_idx_pseudo , 2]
        #     ## figure specification
        #     plot_file_name=f'pseudo_{sim_model}_{t_str}.pdf'
        #     plot_file=os.path.join(plot_dir,plot_file_name)
        #     fig, ax = plt.subplots(figsize=(5, 5))
        #     ## scatter plot
        #     pseudo_pos_x=pseudo_pos_3d[:,:,z_idx_pseudo,0]
        #     pseudo_pos_y=pseudo_pos_3d[:,:,z_idx_pseudo,1]
        #     pseudo_b_xy=pseudo_b_3d[:,:,z_idx_pseudo]
        #     img = ax.scatter(pseudo_pos_x, pseudo_pos_y, c=pseudo_b_xy, s=3, cmap='viridis')
        #     ## color bar
        #     cbar = plt.colorbar(img, ax=ax)  # Add colorbar to subplot 1
        #     cbar_label=r"Scattering length (b) [$10^{-5} \cdot \mathrm{\AA} = \mathrm{fm}$]"
        #     cbar.set_label(cbar_label, labelpad=10)
        #     ## plot title
        #     title_text=" Cut at Z = {0} {1}".format(z_val_pseudo, r"$\mathrm{\AA}$")
        #     ax.set_title(title_text)
        #     # labels
        #     ax.set_xlabel(r'X [$\mathrm{\AA}$]')
        #     ax.set_ylabel(r'Y [$\mathrm{\AA}$]')
        #     # other formatting
        #     ax.set_aspect('equal')
        #     ax.set_xlim([0, length_a])
        #     ax.set_ylim([0, length_b])
        #     ## add mesh
        #     cell_x=length_a/nx
        #     cell_y=length_a/ny
        #     for idx1 in range(nx):
        #         for idx2 in range(ny):
        #             rect_center_x=idx1*cell_x
        #             rect_center_y=idx2*cell_y
        #             ax.add_patch(Rectangle((rect_center_x, rect_center_y), cell_x, cell_y,
        #                                     edgecolor='k', facecolor='none', linewidth=0.5))
        #     plt.savefig(plot_file, format='pdf')
        #     plt.close(fig)

        #     # plotting categorized pseudo atoms
        #     ## figure specification
        #     plot_file_name=f'pseudo_cat_{sim_model}_{t_str}.pdf'
        #     plot_file=os.path.join(plot_dir,plot_file_name)
        #     fig, ax = plt.subplots(figsize=(5, 5))
        #     ## scatter plot
        #     pseudo_b_cat_val_3d=pseudo_b_cat_val.reshape((nx, ny, nz))
        #     pseudo_b_cat_val_xy=pseudo_b_cat_val_3d[:,:,z_idx_pseudo]
        #     img = ax.scatter(pseudo_pos_x, pseudo_pos_y, c=pseudo_b_cat_val_xy, s=3, cmap='viridis')
        #     ## color bar
        #     cbar = plt.colorbar(img, ax=ax)  # Add colorbar to subplot 1
        #     cbar_label=r"Scattering length (b) [$10^{-5} \cdot \mathrm{\AA} = \mathrm{fm}$]"
        #     cbar.set_label(cbar_label, labelpad=10)
        #     ## plot title
        #     title_text=" Cut at Z = {0} {1}".format(z_val_pseudo, r"$\mathrm{\AA}$")
        #     ax.set_title(title_text)
        #     # labels
        #     ax.set_xlabel(r'X [$\mathrm{\AA}$]')
        #     ax.set_ylabel(r'Y [$\mathrm{\AA}$]')
        #     # other formatting
        #     ax.set_aspect('equal')
        #     ax.set_xlim([0, length_a])
        #     ax.set_ylim([0, length_b])
        #     x_ticks_val=np.round(np.linspace(0, length_a, 5),2)
        #     ax.set_xticks(list(x_ticks_val))
        #     y_ticks_val=np.round(np.linspace(0, length_b, 5),2)
        #     ax.set_yticks(list(y_ticks_val))
        #     ## add mesh
        #     cell_x=length_a/nx
        #     cell_y=length_a/ny
        #     for idx1 in range(nx):
        #         for idx2 in range(ny):
        #             rect_center_x=idx1*cell_x
        #             rect_center_y=idx2*cell_y
        #             ax.add_patch(Rectangle((rect_center_x, rect_center_y), cell_x, cell_y,
        #                                     edgecolor='k', facecolor='none', linewidth=0.5))
        #     plt.savefig(plot_file, format='pdf')
        #     plt.close(fig)
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

    # plot scatt pattern for this time step
    plot_file_name=f'Iq_{sim_model}_{t_str}.pdf'
    plot_file=os.path.join(plot_dir,plot_file_name)
    fig, ax = plt.subplots(figsize=(7, 5))
    #
    # loglog plot
    ax.loglog(q_ana, Iq_ana, 'b', label= 'Analytical calculation')
    ax.loglog(q_num, Iq_num, 'r',
               linestyle='',
                 marker='o', markersize=3,
                   label= 'Numerical calculation')
    #
    # plot formatting
    ## legend
    ax.legend()
    ## labels
    ax.set_xlabel(r'Q [$\mathrm{\AA}^{-1}$]')
    ax.set_ylabel(r'I(Q) [$\mathrm{cm}^{-1}$]')
    ## SANS upper boundary Q=1 \AA^-1
    ax.set_xlim([Q_range[0], Q_range[1]])
    # ax.set_ylim(bottom=1e3)
    ## save plot
    plt.savefig(plot_file, format='pdf')
    plt.close(fig)

    # plot fit param upto this time step
    plot_file_name=f'sld_fit_{sim_model}_{t_str}.pdf'
    plot_file=os.path.join(plot_dir,plot_file_name)
    fig, ax = plt.subplots(figsize=(7, 5))
    #
    # plot
    ax.plot(t_arr, sld_ana, 'b', label= 'Simulation value')
    ax.plot(t_arr[0:i+1], sld_fit[0:i+1], 'r', linestyle='', marker='^',
             markersize=5, label= 'Fit value')
    #
    # plot formatting
    ## legend
    ax.legend()
    ## labels
    ax.set_xlabel('Time [{s}]')
    ax.set_ylabel(r'SLD [$10^{-5} \cdot \mathrm{\AA}^{-2}$]')
    ## limits
    #ax.set_xlim([t_arr[0], t_arr[-1]])
    #ax.set_ylim([rad_0,rad_end])
    ax.grid(True)
    ## save plot
    plt.savefig(plot_file, format='pdf')
    plt.close(fig)

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
