"""
This plots necessary figures for phase field model
Plots are saved in figure folder

Author: Arnab Majumdar
Date: 24.06.2025
"""
import sys
import os
lib_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(lib_dir)

import os
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py
from matplotlib.patches import Rectangle
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import cm

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

# analytical SAS function
# non req., only numerical SAS calculated

"""
Input data
"""
### file locations ###
# xml location
xml_folder='./xml/'
# script dir and working dir
script_dir = "src"  # Full path of the script
working_dir = "."  # Directory to run the script in

# phase field simulation details 
phenm='spinodal_fe_cr'
sim_times=(np.array([0, 2, 4, 6, 8, 8.64]) * 10000).astype(int)

# figure initialize
fig_scatt, ax_scatt = plt.subplots(figsize=(7, 5))
fig_ch_len, ax_ch_len = plt.subplots(figsize=(7, 5))

# Choose a colormap
cmap = cm.winter_r  # or 'plasma', 'rainbow', etc.
# Number of colors you want
N = len(sim_times)
# Generate an array of colors (RGBA format)
colors = cmap(np.linspace(0, 1, N))

# initialize fit params (characteristic length)
ch_len_arr=np.zeros_like(sim_times, dtype='float')

for sim_t_idx, sim_time in enumerate(sim_times):
    print(f'current time step: {sim_time}')

    # read moose input file
    moose_inp_file_name='{0}_{1:0>5}.h5'.format(phenm, sim_time)
    moose_inp_file=os.path.join('moose', moose_inp_file_name)
    moose_inp=h5py.File(moose_inp_file,'r')

    ### struct gen ###
    # box side lengths (float values) 
    length_a=moose_inp['length_a'][()].astype('float')
    length_b=moose_inp['length_b'][()].astype('float')
    length_c=moose_inp['length_c'][()].astype('float')
    # number of cells in each direction (int values)
    nx=moose_inp['nx'][()]
    ny=moose_inp['ny'][()]
    nz=moose_inp['ny'][()]
    # element details 
    el_type='lagrangian'
    el_order=1
    # calculate mid point of structure (simulation box)
    mid_point=np.array([length_a/2, length_b/2, length_c/2])

    ### sim gen ###
    sim_model='phase_field'
    dt=1.
    t_end=0.
    n_ensem=1
    # calculate time array
    t_arr=np.arange(0,t_end+dt, dt)

    ### model_param ###
    name=phenm
    moose_time=moose_inp['time'][()]
    qclean_sld=moose_inp['qclean_sld'][()]
    # dir name for model param
    model_param_dir_name = ('name_' + str(name) + 
                            '_time_' + str(moose_time) + 
                            '_qclean_sld_' + str(qclean_sld)
                            ).replace('.','p')

    ### scatt_cal ###
    num_cat=501
    method_cat='extend'
    sassena_exe= '/home/amajumda/Documents/Softwares/sassena/compile/sassena'
    mpi_procs=4
    num_threads=2
    sig_file='signal.h5'
    scan_vec=np.array([1, 0, 0])
    Q_range=np.array([2*np.pi/length_a, 1])
    num_points=100
    num_orientation=200
    # scatt settengs
    scatt_settings='cat_' + method_cat + '_' + str(num_cat) + 'Q_' \
        + str(Q_range[0]) + '_' + str(Q_range[1]) + '_' + 'orien__' + str(num_orientation)
    scatt_settings=scatt_settings.replace('.', 'p')
    
    # close moose input file
    moose_inp.close()

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
          + '_' + nx_str + '_' + ny_str+'_' + nz_str + '_' + el_order_str
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
    plot_dir=os.path.join(figure_dir, sim_model)
    os.makedirs(plot_dir, exist_ok=True)

    for i in range(len(t_arr)):
        t=t_arr[i]
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


            if idx_ensem==0:
                if el_type=='lagrangian':
                    num_node_x=el_order*nx+1
                    num_node_y=el_order*ny+1
                    num_node_z=el_order*nz+1
                # plotting node SLD
                ## cutting at z = cut_frac * length_z
                cut_frac=0.5
                node_pos_3d=node_pos.reshape(num_node_x, num_node_y, num_node_z, 3)
                z_idx= np.floor(cut_frac*(num_node_z)).astype(int)
                z_val=node_pos_3d[0, 0, z_idx , 2]
                ## figure specification
                plot_file_name=f'SLD_phase_field_{moose_time}.pdf'
                plot_file=os.path.join(plot_dir,plot_file_name)
                fig, ax = plt.subplots(figsize=(5, 5))
                ## image plot
                ### .T is required to exchange x and y axis 
                ### origin is 'lower' to put it in lower left corner 
                node_sld_3d=node_sld.reshape(num_node_x, num_node_y, num_node_z)
                img = ax.imshow(node_sld_3d[:,:,z_idx].T, 
                                extent=[0, length_a, 0, length_b], 
                                origin='lower', vmin=sld_min, vmax=sld_max, interpolation='bilinear')
                ## color bar
                cbar = plt.colorbar(img, ax=ax)  # Add colorbar to subplot 1
                cbar_label=r"Scattering length density (SLD) [$10^{-5} \cdot \mathrm{\AA}^{-2}$]"
                cbar.set_label(cbar_label, labelpad=10)
                ## plot title
                title_text=" Cut at Z = {0} {1}".format(z_val, r"$\mathrm{\AA}$")
                ax.set_title(title_text)
                # labels
                ax.set_xlabel(r'X [$\mathrm{\AA}$]')
                ax.set_ylabel(r'Y [$\mathrm{\AA}$]')
                ## add mesh
                cell_x=length_a/nx
                cell_y=length_a/ny
                for idx1 in range(nx):
                    for idx2 in range(ny):
                        rect_center_x=idx1*cell_x
                        rect_center_y=idx2*cell_y
                        ax.add_patch(Rectangle((rect_center_x, rect_center_y), cell_x, cell_y, 
                                            edgecolor='k', facecolor='none', linewidth=0.5))


                # save figure
                plt.savefig(plot_file, format='pdf')
                plt.close(fig)

                # plotting pseudo atoms
                ## cutting at z = z_val + length_z
                # cell_z= length_c/nz
                # z_val_pseudo=z_val+cell_z
                cut_frac=0.5
                pseudo_pos_3d=pseudo_pos.reshape(nx, ny, nz, 3)
                pseudo_b_3d=pseudo_b.reshape(nx, ny, nz)
                z_idx_pseudo= nz//2-1 #z_idx-1
                z_val_pseudo=pseudo_pos_3d[0, 0, z_idx_pseudo , 2]
                ## figure specification
                plot_file_name=f'pseudo_phase_field_{moose_time}.pdf'
                plot_file=os.path.join(plot_dir,plot_file_name)
                fig, ax = plt.subplots(figsize=(5, 5))
                ## scatter plot
                pseudo_pos_x=pseudo_pos_3d[:,:,z_idx_pseudo,0]
                pseudo_pos_y=pseudo_pos_3d[:,:,z_idx_pseudo,1]
                pseudo_b_xy=pseudo_b_3d[:,:,z_idx_pseudo]
                img = ax.scatter(pseudo_pos_x, pseudo_pos_y, c=pseudo_b_xy, s=3, cmap='viridis')
                ## color bar
                cbar = plt.colorbar(img, ax=ax)  # Add colorbar to subplot 1
                cbar_label=r"Scattering length (b) [$10^{-5} \cdot \mathrm{\AA} = \mathrm{fm}$]"
                cbar.set_label(cbar_label, labelpad=10)
                ## plot title
                title_text=" Cut at Z = {0} {1}".format(z_val_pseudo, r"$\mathrm{\AA}$")
                ax.set_title(title_text)
                # labels
                ax.set_xlabel(r'X [$\mathrm{\AA}$]')
                ax.set_ylabel(r'Y [$\mathrm{\AA}$]')
                # other formatting
                ax.set_aspect('equal')
                ax.set_xlim([0, length_a])
                ax.set_ylim([0, length_b])

                ## add mesh
                cell_x=length_a/nx
                cell_y=length_a/ny
                for idx1 in range(nx):
                    for idx2 in range(ny):
                        rect_center_x=idx1*cell_x
                        rect_center_y=idx2*cell_y
                        ax.add_patch(Rectangle((rect_center_x, rect_center_y), cell_x, cell_y, 
                                            edgecolor='k', facecolor='none', linewidth=0.5))
                plt.savefig(plot_file, format='pdf')
                plt.close(fig)

                # plotting categorized pseudo atoms
                ## figure specification
                plot_file_name=f'pseudo_cat_phase_field_{moose_time}.pdf'
                plot_file=os.path.join(plot_dir,plot_file_name)
                fig, ax = plt.subplots(figsize=(5, 5))
                ## scatter plot
                pseudo_b_cat_val_3d=pseudo_b_cat_val.reshape(nx, ny, nz)
                pseudo_b_cat_val_xy=pseudo_b_cat_val_3d[:,:,z_idx_pseudo]
                img = ax.scatter(pseudo_pos_x, pseudo_pos_y, c=pseudo_b_cat_val_xy, s=5, cmap='viridis')
                ## color bar
                cbar=plt.colorbar(img, ax=ax)  # Add colorbar to subplot 1
                cbar_label=r"Scattering length (b) [$10^{-5} \cdot \mathrm{\AA} = \mathrm{fm}$]"
                cbar.set_label(cbar_label, labelpad=10)
                ## plot title
                title_text=" Cut at Z = {0} {1}".format(z_val_pseudo, r"$\mathrm{\AA}$")
                ax.set_title(title_text)
                # labels
                ax.set_xlabel(r'X [$\mathrm{\AA}$]')
                ax.set_ylabel(r'Y [$\mathrm{\AA}$]')
                # other formatting
                ax.set_aspect('equal')
                ax.set_xlim([0, length_a])
                ax.set_ylim([0, length_b])
                x_ticks_val=np.round(np.linspace(0, length_a, 5),2)
                ax.set_xticks(list(x_ticks_val))
                y_ticks_val=np.round(np.linspace(0, length_b, 5),2)
                ax.set_yticks(list(y_ticks_val))
                
                # cut aspet inset
                x_ticks_val=np.round(np.linspace(100, 200, 6),2)
                ax.set_xticks(list(x_ticks_val))
                y_ticks_val=np.round(np.linspace(100, 200, 6),2)
                ax.set_yticks(list(y_ticks_val))
                ## add mesh
                cell_x=length_a/nx
                cell_y=length_a/ny
                for idx1 in range(nx):
                    for idx2 in range(ny):
                        rect_center_x=idx1*cell_x
                        rect_center_y=idx2*cell_y
                        ax.add_patch(Rectangle((rect_center_x, rect_center_y), cell_x, cell_y, 
                                            edgecolor='k', facecolor='none', linewidth=0.5))
                plt.savefig(plot_file, format='pdf')
                plt.close(fig)

        # volume for normalization
        vol_norm=length_a*length_b*length_c

        # numerical intensity
        ## read I vs Q signal file
        Iq_data_file_name='Iq_{0}.h5'.format(scatt_settings) 
        Iq_data_file=os.path.join(t_dir,Iq_data_file_name)
        Iq_data=h5py.File(Iq_data_file,'r')
        Iq=Iq_data['Iq'][:] # unit fm^2
        Iq=Iq/vol_norm*10**2 # unit (fm^2 / \AA^3) * 10^2 = cm^-1
        q=Iq_data['Q'][:]
        Iq_data.close()
        
        
        # plot scattring pattern
        ax_scatt.semilogx(q, Iq,color=colors[sim_t_idx], 
                    linestyle='-', marker='none', markersize=3, 
                    label=f't = {moose_time} s')
        
        # position of peak position
        # find Q_p
        Iq_p=max(Iq)
        Iq_p_arg=np.argmax(Iq)
        q_p=q[Iq_p_arg]
        # calculate characteristic length
        ch_len=2*np.pi/q_p 
        ch_len=round(ch_len,2) # in \AA
        ch_len_nm=ch_len/10 
        ch_len_nm=round(ch_len_nm,2) # in nm
        # print characteristic length
        AA_code = "\u212B" 
        print(f'Characteristic length is {ch_len} {AA_code} ({ch_len_nm} nm)')
        # store ch len in array
        ch_len_arr[sim_t_idx]=ch_len_nm
        

# finalize plot for scattering
# plot formatting
## legend
ax_scatt.legend()
## labels
ax_scatt.set_xlabel(r'Q [$\mathrm{\AA}^{-1}$]')
ax_scatt.set_ylabel(r'I(Q) [$\mathrm{cm}^{-1}$]')
## SANS upper boundary Q=1 \AA^-1
ax_scatt.set_xlim([Q_range[0], Q_range[1]])
## grid
ax_scatt.grid()
## save plot for scattering
plot_file=os.path.join(plot_dir, f'Iq.pdf')
fig_scatt.savefig(plot_file, format='pdf')
plt.close(fig_scatt)

# finalize plot for characteristic length
# plot scattring pattern
ax_ch_len.plot(sim_times[1:], ch_len_arr[1:], 
            linestyle='--', marker='o', markersize=5,
            label=phenm)
# plot formatting
## legend
ax_ch_len.legend()
## labels
ax_ch_len.set_xlabel('Simulation time [s]')
ax_ch_len.set_ylabel('characteristic length [nm]')
## grid
ax_ch_len.grid()
## save plot for scattering
plot_file=os.path.join(plot_dir, f'ch_len.pdf')
fig_ch_len.savefig(plot_file, format='pdf')
plt.close(fig_scatt)
