"""
This plots figure used in publication for grain growth model
Plots are saved in figure folder

Author: Arnab Majumdar
Date: 24.06.2025
"""
import sys
import os
lib_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(lib_dir)

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py
from matplotlib.patches import Rectangle
from scipy.optimize import curve_fit
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ignore warnings
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

# analytical SAS function
def J1(x):
    if x==0:
        return 0
    else:
        return (np.sin(x)-x*np.cos(x))/x**2

def ball (qmax,qmin,Npts,scale,bg,sld,sld_sol,rad):
    vol=(4/3)*np.pi*rad**3
    # SLD unit 10^-5 \AA^-2
    del_rho=sld-sld_sol
    q_arr=np.linspace(qmin,qmax,Npts)
    FormFactor=np.zeros(len(q_arr))
    for i in range(len(q_arr)):
        q=q_arr[i]
        if q==0:
            # Form factor unit 10^-5 \AA
            FormFactor[i]=vol*del_rho
        else:
            # Form factor unit 10^-5 \AA
            FormFactor[i]=3*vol*del_rho*J1(q*rad)/(q*rad)
    # Intensity unit 10^-10 \AA^2
    Iq_arr = ((scale)*np.abs(FormFactor)**2+bg)
    return Iq_arr, q_arr

def fit_func_gg(q_in, rad_opt):
    Iq, q_out =ball(qmax=max(q_in),qmin=min(q_in),
                        Npts=len(q_in), scale=1 , bg=0,
                        sld=sld_in, sld_sol=sld_out,
                        rad=rad_opt)
    return Iq

"""
read input from xml file
"""

### file locations ###
# xml location
xml_folder='./xml/'
# script dir and working dir
script_dir = "src"  # Full path of the script
working_dir = "."  # Directory to run the script in

"""
Input data
"""
### struct gen ###
# box side lengths (float values)
length_a=40. 
length_b=40. 
length_c=40.
# number of cells in each direction (int values)
nx=40 
ny=40 
nz=40 
# element details
el_type='lagrangian'
el_order=1
# calculate mid point of structure (simulation box)
mid_point=np.array([length_a/2, length_b/2, length_c/2])

### sim_gen ###
sim_model='gg'
# simulation parameters
dt=1.
t_end=10.
n_ensem=1
# calculate time array
t_arr=np.arange(0,t_end+dt, dt)

### model param ###
# model params
rad_0=7
rad_end=17
sld_in=2
sld_out=1
qclean_sld=sld_out
# dir name for model param
model_param_dir_name = ('rad_0' + '_' + str(rad_0) + '_' +
                        'rad_end' + '_' + str(rad_end) + '_' +
                        'sld_in' + '_' + str(sld_in) + '_' +
                        'sld_out' + '_' + str(sld_out) + '_' +
                        'qclean_sld' + '_' + str(qclean_sld)).replace('.', 'p')

### scatt_cal ###
# decreitization params
# number of categories and method of categorization
num_cat=3
method_cat='extend'
sig_file='signal.h5'
scan_vec=np.array([1, 0, 0])
Q_range=np.array([0., 1.])
num_points=100
num_orientation=10
# scatt settengs
scatt_settings='cat_' + method_cat + '_' + str(num_cat) + 'Q_' \
    + str(Q_range[0]) + '_' + str(Q_range[1]) + '_' + 'orien__' + str(num_orientation)
scatt_settings=scatt_settings.replace('.', 'p')

"""
read folder structure
"""

# """
# calculate vars and create folder structure
# """

# xml_folder=xml_dir

# ### struct xml ###

# # calculate mid point of structure (simulation box)
# mid_point=np.array([length_a/2, length_b/2, length_c/2])

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

# ### sim xml entries ###

# # time array
# t_arr=np.arange(0,t_end+dt, dt)

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
plot_dir=os.path.join(figure_dir, sim_model + '_paper')
os.makedirs(plot_dir, exist_ok=True)
# initialize figures (1: SAS pattern, 2: Fit param)
fig_scatt_all, ax_scatt_all = plt.subplots(figsize=(7, 5))
fig_fit_all, ax_fit_all = plt.subplots(figsize=(7, 5))
# color scheme
color_rainbow = plt.cm.rainbow(np.linspace(0, 1, len(t_arr)))

# initialize fit_params (radius)
rad_fit=np.zeros_like(t_arr)
rad_ana=np.zeros_like(t_arr)

for i in range(len(t_arr)):
    t=t_arr[i]
    t_str=str(t).replace('.','p')
    # time_dir name
    t_dir_name='t{0:0>3}'.format(i)
    t_dir=os.path.join(model_param_dir, t_dir_name)
    print(f'plotting fot time: {t/t_end} [t/tmax]')
    for j in range(n_ensem):
        idx_ensem=j
        # create ensemble dir
        ensem_dir_name='ensem{0:0>3}'.format(idx_ensem)
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
        node_sld=scatt_cal_data['node_sld'][:]
        pseudo_pos=scatt_cal_data['pseudo_pos'][:]
        pseudo_b=scatt_cal_data['pseudo_b'][:]
        pseudo_b_cat_val=scatt_cal_data['pseudo_b_cat_val'][:]
        pseudo_b_cat_idx=scatt_cal_data['pseudo_b_cat_idx'][:]
        scatt_cal_data.close()

        # determine sld min and max for plotting
        sld_min=np.min(node_sld,0)
        sld_max=np.max(node_sld,0)


        if idx_ensem==0:
            if el_type=='lagrangian':
                num_node_x=el_order*nx+1
                num_node_y=el_order*ny+1
                num_node_z=el_order*nz+1
            # plotting node SLD
            ## cutting at z = cut_frac * length_z
            cut_frac=0.5
            node_pos_3d=node_pos.reshape(num_node_x, num_node_y, num_node_z, 3)
            z_idx= np.floor(cut_frac*(nz+1)).astype(int)
            z_val=node_pos_3d[0, 0, z_idx , 2]
            # image plot
            ## .T is required to exchange x and y axis 
            ## origin is 'lower' to put it in lower left corner 
            node_sld_3d=node_sld.reshape(nx+1, ny+1, nz+1)
            
            # inset in the final figure
            if i==0:
                ax_ins = inset_axes(ax_fit_all, width="100%", height="100%", bbox_to_anchor=(0.15, 0.55, 0.3, 0.3),
                        bbox_transform=ax_fit_all.transAxes)
                img = ax_ins.imshow(node_sld_3d[:,:,z_idx].T, 
                                extent=[0, length_a, 0, length_b], 
                                origin='lower', vmin=sld_min, vmax=sld_max, interpolation='bilinear')
                ## add mesh
                cell_x=length_a/nx
                cell_y=length_a/ny
                for idx1 in range(nx):
                    for idx2 in range(ny):
                        rect_center_x=idx1*cell_x
                        rect_center_y=idx2*cell_y
                        ax_ins.add_patch(Rectangle((rect_center_x, rect_center_y), cell_x, cell_y, 
                                            edgecolor='none', facecolor='none', linewidth=0.5))
                ax_ins.axis('off')
                
                ## Draw arrow
                # Draw a vertical line (upward)
                ax_fit_all.plot([0, 0], [7, 14], linewidth=1, color='black', linestyle='--')
                # Draw a horizontal line (to the right)
                ax_fit_all.plot([0, 2], [14, 14], linewidth=1, color='black', linestyle='--')
                # Draw arrowhead
                ax_fit_all.arrow(1.2, 14, 0.001, 0, linewidth=1, head_width=0.5, head_length=0.3, 
                                 fc='black', ec='black')
                                

            elif i==len(t_arr)-1:
                ax_ins = inset_axes(ax_fit_all, width="100%", height="100%", bbox_to_anchor=(0.6, 0.1, 0.3, 0.3),
                        bbox_transform=ax_fit_all.transAxes)
                
                img = ax_ins.imshow(node_sld_3d[:,:,z_idx].T, 
                                extent=[0, length_a, 0, length_b], 
                                origin='lower', vmin=sld_min, vmax=sld_max, interpolation='bilinear')
                ## add mesh
                cell_x=length_a/nx
                cell_y=length_a/ny
                for idx1 in range(nx):
                    for idx2 in range(ny):
                        rect_center_x=idx1*cell_x
                        rect_center_y=idx2*cell_y
                        ax_ins.add_patch(Rectangle((rect_center_x, rect_center_y), cell_x, cell_y, 
                                            edgecolor='none', facecolor='none', linewidth=0.5))
                ax_ins.axis('off')

                ## Draw arrow
                # Draw a vertical line (upward)
                ax_fit_all.plot([10, 10], [17, 9], linewidth=1, color='black', linestyle='--')
                # Draw a horizontal line (to the right)
                ax_fit_all.plot([10, 8], [9, 9], linewidth=1, color='black', linestyle='--')
                # Draw arrowhead
                ax_fit_all.arrow(9.1, 9, -0.001, 0, linewidth=1, head_width=0.5, head_length=0.3, 
                                 fc='black', ec='black')
    
    # box geometry
    vol_box=length_a*length_b*length_c

    # volume for normalization
    vol_norm=vol_box

    # numerical intensity
    ## read I vs Q signal file
    Iq_data_file_name='Iq_{0}.h5'.format(scatt_settings) 
    Iq_data_file=os.path.join(t_dir,Iq_data_file_name)
    Iq_data=h5py.File(Iq_data_file,'r')
    Iq_num_raw=Iq_data['Iq'][:] # unit fm^2
    Iq_num=Iq_num_raw/vol_norm*10**2 # unit (fm^2 / \AA^3) * 10^2 = cm^-1
    q_num=Iq_data['Q'][:]
    Iq_data.close()

    # fit radius w.r.t. numerical intensity
    popt, pcov = curve_fit(fit_func_gg, q_num, Iq_num_raw)
    rad_fit[i]=popt[0]

    # ananlytical intensity 
    ## Intensity unit 10^-10 \AA^2
    Iq_ana_raw,q_ana= ball(qmax=np.max(q_num),qmin=np.min(q_num),Npts=100,
                scale=1,bg=0,sld=sld_in, sld_sol=sld_out, rad=rad_fit[i])
    
    ## Normalize by volume
    ## (Before * 10**2) Intensity unit 10^-10 \AA^-1 = 10 ^-2 cm^-1
    ## (after * 10**2) Intensity unit cm^-1
    Iq_ana = (Iq_ana_raw / vol_norm) * 10**2


# plot fit param for all time step
plot_file_name='rad_fit_{0}.pdf'.format(sim_model)
plot_file=os.path.join(plot_dir,plot_file_name)
# fig, ax = plt.subplots(figsize=(7, 5))

rad_ana=rad_0+t_arr*(rad_end-rad_0)/t_end
ax_fit_all.plot(t_arr, rad_fit, 'r', linestyle='', marker='^', markersize=5, label= 'Retrieved value')
ax_fit_all.plot(t_arr, rad_ana, 'gray', zorder=-10, label= 'Simulation value')


# plot formatting
## legend
ax_fit_all.legend()
## labels
ax_fit_all.set_xlabel('Time [s]')
ax_fit_all.set_ylabel(r'Radius of grain [$\mathrm{\AA}$]')
## limits
ax_fit_all.grid(True)
## save plot
plt.savefig(plot_file, format='pdf')
plt.close('all')