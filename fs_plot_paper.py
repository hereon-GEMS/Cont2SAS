"""
This plots figure used in publication for interdiffusion model
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

def fuzzysph(qmax,qmin,Npts,scale,bg,sld,sld_sol,sig_fuzz,radius):
    vol=(4/3)*np.pi*radius**3
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
            FormFactor[i]=(3*vol*del_rho*J1(q*radius)/(q*radius))*np.exp((-(sig_fuzz*q)**2)/2)
    # Intensity unit 10^-10 \AA^2
    Iq_arr = scale*np.abs(FormFactor)**2+bg
    return Iq_arr, q_arr

def fit_func(q_in, sig_opt, rad_opt):
    Iq, q_out =fuzzysph(qmax=max(q_in),qmin=min(q_in),
                        Npts=len(q_in), scale=1 , bg=0,
                        sld=sld_in, sld_sol=sld_out,
                        sig_fuzz=sig_opt, radius=rad_opt)
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
# mesh details
nx_arr=[50,100,50]
el_order_arr=[1,1,2]
# figure initialize
fig1, ax1 = plt.subplots(figsize=(7, 5))
fig2, ax2 = plt.subplots(figsize=(7, 5))
fig3, ax3 = plt.subplots(2,figsize=(7, 5), gridspec_kw={'hspace': 0.1, 'height_ratios': [3, 1]})
# plot settings
marker=['s', '^', 'o']
colors=['lime', 'm', 'yellow']
ms=[6,6,4]

for vary_idx in range(len(nx_arr)):
    ### struct gen ###
    # box side lengths (float values)
    length_a=200. 
    length_b=length_a 
    length_c=length_a
    # number of cells in each direction (int values)
    nx=nx_arr[vary_idx] 
    ny=nx 
    nz=nx 
    # element details
    el_type='lagrangian'
    el_order=el_order_arr[vary_idx]
    # calculate mid point of structure (simulation box)
    mid_point=np.array([length_a/2, length_b/2, length_c/2])
    # #calculate cell lengths
    # cell_x=length_a/nx
    # cell_y=length_a/nx
    # cell_z=length_a/nx

    ### sim gen ###
    sim_model='fs'
    dt=1.
    t_end=10.
    n_ensem=1
    # calculate time array
    t_arr=np.arange(0,t_end+dt, dt)

    ### model_param ###
    rad=60
    sig_0=2
    sig_end=10
    sld_in=5
    sld_out=1
    qclean_sld=sld_out
    # dir name for model param
    model_param_dir_name = ('rad' + '_' + str(rad) + '_' +
                            'sig_0' + '_' + str(sig_0) + '_' +
                            'sig_end' + '_' + str(sig_end) + '_' +
                            'sld_in' + '_' + str(sld_in) + '_' +
                            'sld_out' + '_' + str(sld_out) + '_' + 
                            'qclean_sld' + '_' + str(qclean_sld)
                            ).replace('.', 'p')


    ### scatt_cal ###
    num_cat=501
    method_cat='extend'
    sig_file='signal.h5'
    scan_vec=np.array([1, 0, 0])
    Q_range=np.array([0., 0.2])
    num_points=100
    num_orientation=100
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
    plot_dir=os.path.join(figure_dir, sim_model + '_paper')
    os.makedirs(plot_dir, exist_ok=True)
    # color scheme
    color_rainbow = plt.cm.rainbow(np.linspace(0, 1, len(t_arr)))

    # initialize fit_params (radius, fuzz value)
    rad_fit=np.zeros_like(t_arr)
    rad_ana=np.ones_like(t_arr)*rad
    sig_fit=np.zeros_like(t_arr)
    sig_ana=sig_0+t_arr*(sig_end-sig_0)/t_end

    for i in range(len(t_arr)):
        t=t_arr[i]
        t_str=str(t).replace('.','p')
        # time_dir name
        t_dir_name='t{0:0>3}'.format(i)
        t_dir=os.path.join(model_param_dir, t_dir_name)
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

            if vary_idx==0 and idx_ensem==0:
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
                ## figure specification
                plot_file_name='SLD_{0}_{1}.pdf'.format(sim_model, t_str)
                plot_file=os.path.join(plot_dir,plot_file_name)
                fig, ax = plt.subplots(figsize=(5, 5))
                ## image plot
                ### .T is required to exchange x and y axis 
                ### origin is 'lower' to put it in lower left corner 
                node_sld_3d=node_sld.reshape(nx+1, ny+1, nz+1)
                
                # inset in the final figure
                if i==0:
                    ax_ins = inset_axes(ax3[0], width="100%", height="100%", bbox_to_anchor=(0.22, -0.2, 0.56, 0.56),
                            bbox_transform=ax3[0].transAxes)
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

                    

                elif i==len(t_arr)-1:
                    ax_ins = inset_axes(ax3[0], width="100%", height="100%", bbox_to_anchor=(0.51, -0.2, 0.56, 0.56),
                            bbox_transform=ax3[0].transAxes)
                    img = ax_ins.imshow(node_sld_3d[:,:,z_idx].T, 
                                    extent=[0, length_a, 0, length_b], 
                                    origin='lower', vmin=sld_min, vmax=sld_max, interpolation='bilinear')
                    ax_ins.axis('off')
                    ## add mesh
                    cell_x=length_a/nx
                    cell_y=length_a/ny
                    for idx1 in range(nx):
                        for idx2 in range(ny):
                            rect_center_x=idx1*cell_x
                            rect_center_y=idx2*cell_y
                            ax_ins.add_patch(Rectangle((rect_center_x, rect_center_y), cell_x, cell_y, 
                                                edgecolor='none', facecolor='none', linewidth=0.5))

        
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

        # fit radius and sig (fuzz_value) w.r.t. numerical intensity
        popt, pcov = curve_fit(fit_func, q_num, Iq_num_raw, 
                            bounds=([sig_0, rad-cell_x], [sig_end, rad+cell_x]))
        sig_fit[i]=round(np.abs(popt[0]),2)
        rad_fit[i]=round(popt[1],2)

    ax1.plot(t_arr, sig_fit, linestyle='', marker=marker[vary_idx], 
             color=colors[vary_idx], markersize=ms[vary_idx], markeredgecolor='k', markeredgewidth=0.5,
             label= 'Meshing: {0} elements of order {1}'.format(nx*ny*nz, el_order))
    ax3[0].plot(t_arr, sig_fit, linestyle='', marker=marker[vary_idx], 
             color=colors[vary_idx], markersize=ms[vary_idx], markeredgecolor='k', markeredgewidth=0.5,
             label= 'Meshing: {0} elements of order {1}'.format(nx*ny*nz, el_order))
    if vary_idx==0:
        ## Draw arrow (t=0)
        # Draw a horrizontal line (rightward)
        ax3[0].plot([0, 0.5], [sig_fit[0], sig_fit[0]], linewidth=1, color='black', linestyle='--')
        # Draw a vertical line (downward)
        ax3[0].plot([0.5, 0.5], [sig_fit[0], 2], linewidth=1, color='black', linestyle='--')
        # Draw a vertical line (rightward)
        ax3[0].plot([0.5, 3.5], [2, 2], linewidth=1, color='black', linestyle='--')
        # Draw arrowhead
        ax3[0].arrow(3, 2, 0.001, 0, linewidth=1, head_width=0.5, head_length=0.3, 
                            fc='black', ec='black')
        
        ## Draw arrow (t=end)
        # Draw a vertical line (downward)
        ax3[0].plot([10, 10], [sig_fit[-1], 2], linewidth=1, color='black', linestyle='--')
        # Draw a horizontal line (leftward)
        ax3[0].plot([10, 9.5], [2, 2], linewidth=1, color='black', linestyle='--')
        # Draw arrowhead
        ax3[0].arrow(9.9, 2, -0.001, 0, linewidth=1, head_width=0.5, head_length=0.3, 
                            fc='black', ec='black')
    
    
    ax2.plot(t_arr, rad_fit, linestyle='', marker=marker[vary_idx], 
             color=colors[vary_idx], markersize=ms[vary_idx], markeredgecolor='k', markeredgewidth=0.5,
             label= 'Meshing: {0} elements of order {1}'.format(nx*ny*nz, el_order))
    ax3[1].plot(t_arr, rad_fit, linestyle='', marker=marker[vary_idx], 
             color=colors[vary_idx], markersize=ms[vary_idx], markeredgecolor='k', markeredgewidth=0.5,
             label= 'Meshing: {0} elements of order {1}'.format(nx*ny*nz, el_order))
    
    if vary_idx==0:
        ## Draw arrow (t=0)
        # Draw a vertical line (upward)
        ax3[1].plot([0, 0], [rad_fit[0], 64], linewidth=1, color='black', linestyle='--')
        # Draw a vertical line (rightward)
        ax3[1].plot([0, 3.5], [64, 64], linewidth=1, color='black', linestyle='--')
        # Draw arrowhead
        ax3[1].arrow(3, 64, 0.001, 0, linewidth=1, head_width=0.5, head_length=0.3, 
                            fc='black', ec='black')
        
        ## Draw arrow (t=end)
        # Draw a vertical line (downward)
        ax3[1].plot([10, 10], [rad_fit[-1], 64], linewidth=1, color='black', linestyle='--')
        # Draw a horizontal line (leftward)
        ax3[1].plot([10, 9.5], [64, 64], linewidth=1, color='black', linestyle='--')
        # Draw arrowhead
        ax3[1].arrow(9.9, 64, -0.001, 0, linewidth=1, head_width=0.5, head_length=0.3, 
                            fc='black', ec='black')


ax1.plot(t_arr, sig_ana, 'gray', zorder=-10, label= 'Simulation value')
# plot formatting
## legend
ax1.legend()
## labels
ax1.set_xlabel('Time [s]')
ax1.set_ylabel(r'Fuzzyness [$\mathrm{\AA}$]')
## limits
#ax1.grid(True)
plot_file_name='sig_fit_{0}.pdf'.format(sim_model)
plot_file=os.path.join(plot_dir,plot_file_name)
fig1.savefig(plot_file, format='pdf')


ax2.plot(t_arr, rad_ana, 'gray', zorder=-10, label= 'Simulation value')
# plot formatting
## legend
ax2.legend(loc='upper left')
## labels
ax2.set_xlabel('Time [s]')
ax2.set_ylabel(r'Radius of grain [$\mathrm{\AA}$]')
## limits
ax2.set_ylim([rad-4,rad+4])
#ax2.grid(True)
plot_file_name='rad_fit_{0}.pdf'.format(sim_model)
plot_file=os.path.join(plot_dir,plot_file_name)
fig2.savefig(plot_file, format='pdf')

# zusammen plot
ax3[0].plot(t_arr, sig_ana, 'gray', zorder=-10, label= 'Simulation value')
ax3[1].plot(t_arr, rad_ana, 'gray', zorder=-10, label= 'Simulation value')

# top plot formatting
## legend
ax3[0].legend(loc='upper left')
## labels
# ax3[0].set_xlabel('Time [s]')
ax3[0].set_xticklabels([])
ax3[0].set_ylabel(r'Fuzzyness [$\mathrm{\AA}$]')
## limits
ax3[0].grid(True)

# bottom plot formatting
# ## legend
# ax3[0].legend(loc='upper left')
## labels
ax3[1].set_xlabel('Time [s]')
ax3[1].set_ylabel(r'Radius of grain [$\mathrm{\AA}$]')
## limits
ax3[1].set_ylim([rad-1,rad+5])
ax3[1].grid(True)

plot_file_name='all_fit_{0}.pdf'.format(sim_model)
plot_file=os.path.join(plot_dir,plot_file_name)
fig3.savefig(plot_file, format='pdf', )