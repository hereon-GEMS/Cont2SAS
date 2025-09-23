"""
This plots IvsQ figures for phase field model used in publication
Plots are saved in figure folder

Author: Arnab Majumdar
Date: 24.06.2025
"""
import sys
import os

lib_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(lib_dir)

import numpy as np
from math import log
import matplotlib.pyplot as plt
import h5py
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import pyplot as plt
import numpy as np
import fitz
from matplotlib import cm
from scipy.optimize import curve_fit


# ignore warnings
import warnings
warnings.filterwarnings("ignore")

# analytical SAS function
def eq_line(x, m, c):
    """
    equation of a line
    """
    return m * x + c

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

# initialize fit params (characteristic length and error bar)
ch_len_arr=np.zeros_like(sim_times, dtype='float')
ch_len_err_arr=np.zeros_like(sim_times, dtype='float')

for sim_t_idx, sim_time in enumerate(sim_times):
    print(f'current time step: {sim_time}')
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
    sig_file='signal.h5'
    scan_vec=np.array([1, 0, 0])
    Q_range=np.array([2*np.pi/250, 1])
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
    plot_dir=os.path.join(figure_dir, sim_model + '_paper')
    os.makedirs(plot_dir, exist_ok=True)
    ## file for output
    report_file=os.path.join(plot_dir, 'report.txt')

    for i in range(len(t_arr)):
        t=t_arr[i]
        # time_dir name
        t_dir_name='t{0:0>3}'.format(i)
        t_dir=os.path.join(model_param_dir, t_dir_name)

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
        
        # plot
        ax_scatt.semilogx(q, Iq,color=colors[sim_t_idx], 
                          linestyle='-', marker='none', markersize=3, 
                          label='Numerical calculation')
        # add arrows mentioning time
        ax_scatt.text(0.04, 7000+sim_t_idx*3000, 
                f"t = {moose_time}s", 
                fontsize=12, ha='center', va='center',
                bbox=dict(facecolor='yellow', edgecolor='black', boxstyle='round,pad=0.3', alpha=1))
        ## Draw arrow
        # Draw a horizontal line (rightward)
        ax_scatt.plot([0.04, 0.07], [7000+sim_t_idx*3000, 7000+sim_t_idx*3000], linewidth=1, color='black', linestyle='-')
        # # Draw a horizontal line (to the right)
        # Draw arrowhead
        arw_x=0.07
        arw_y=7000+sim_t_idx*3000
        dx=0.1-arw_x
        # Find index of nearest y
        Iq_idx = np.abs(q - 0.1).argmin()
        Iq_target = Iq[Iq_idx]
        q_target = q[Iq_idx]
        dy=Iq_target-arw_y
        # draw arrow
        ax_scatt.plot([0.07, q_target], [7000+sim_t_idx*3000, Iq_target], linewidth=1, color='black', linestyle='-')

        
        # position of peak position
        # find Q_p
        Iq_p=max(Iq)
        Iq_p_arg=np.argmax(Iq)
        q_p=q[Iq_p_arg]
        q_p_err_low=q[Iq_p_arg-1]
        q_p_err_high=q[Iq_p_arg+1]
        # print(f'Qp :{q_p}, Q values before: {q_p_err_low} and after: {q_p_err_high}')
        # calculate characteristic length
        ch_len=2*np.pi/q_p
        ch_len_err=2*np.pi*(1/q_p_err_low - 1/q_p_err_high)
        print(f'errors are : {ch_len_err}')
        ch_len=round(ch_len,2) # in \AA
        ch_len_err=round(ch_len_err,2) # in \AA
        ch_len_nm=ch_len/10
        ch_len_err_nm=ch_len_err/10
        ch_len_nm=round(ch_len_nm,2) # in nm
        ch_len_err_nm=round(ch_len_err_nm,2) # in nm
        # print characteristic length
        AA_code = "\u212B" 
        with open(report_file, "a") as f:
            print(f'Characteristic length is {ch_len} {AA_code} ({ch_len_nm} nm)', file=f)
        # print(f'Characteristic length is {ch_len} {AA_code} ({ch_len_nm} nm)')
        # store ch len and error bar in array
        ch_len_arr[sim_t_idx]=ch_len_nm
        ch_len_err_arr[sim_t_idx]=ch_len_err_nm
# add pdf in the inset axes
ax_ins1 = inset_axes(ax_scatt, width="90%", height="90%", 
                        bbox_to_anchor=(0.41, 0.3, 0.7, 0.7), 
                        bbox_transform=ax_scatt.transAxes)
ax_ins1.axis('off')
# Load the PDF and select the first page
doc = fitz.open('./ppt/phase_field.pdf')
page = doc.load_page(0)
# Define crop area (x0, y0, x1, y1) in points
crop_rect = fitz.Rect(220, 35, 550, 400)  # adjust as needed
# Render the cropped area to a pixmap (image)
pix = page.get_pixmap(clip=crop_rect, dpi=200)
img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
# Plot with matplotlib
ax_ins1.imshow(img)


# plot formatting
## labels
ax_scatt.set_xlabel(r'Q [$\mathrm{\AA}^{-1}$]')
ax_scatt.set_ylabel(r'I(Q) [$\mathrm{cm}^{-1}$]')
## SANS upper boundary Q=1 \AA^-1
ax_scatt.set_xlim([Q_range[0], Q_range[1]])
## grid
ax_scatt.grid()

## save plot
plot_file=os.path.join(plot_dir, 'spinod_fe_cr.pdf')
fig_scatt.savefig(plot_file, format='pdf')
plt.close(fig_scatt)

# plot characteristic length vs time
# ax_ch_len.loglog(sim_times[1:], ch_len_arr[1:],
#                     linestyle='',
#                       markersize=10, marker='*',
#                         label= 'Simulation value')
ax_ch_len.errorbar(sim_times[1:], ch_len_arr[1:],
                    yerr=ch_len_err_arr[1:],
                      fmt='ob', ecolor='gray', capsize=5, markersize=10,
                        label= 'Simulation value')

## fit to a line
### ln(lambda) = a * ln(t) + c
popt, pcov = curve_fit(eq_line, np.log(sim_times[1:]), np.log(ch_len_arr[1:]))
slope=round(popt[0],2)
print(f'slope: {slope}')
log_ch_len_fit=eq_line(np.log(sim_times[1:]), *popt)
ch_len_fit=np.exp(log_ch_len_fit)
ax_ch_len.plot(sim_times[1:], ch_len_fit,
                    linewidth=1, color='k', linestyle='--',
                      label= 'Fitted power law')

### add slope value ad text
# ax_ch_len.text(2e4, 5, f"Slope = {slope}", color='k', fontsize=12,
#                 bbox=dict(facecolor='white', edgecolor='black'))
# Add annotation with arrow
arr_head_x=3e4
arr_head_y=np.exp(eq_line(np.log(arr_head_x), *popt))
print(arr_head_y)
# np.exp(eq_line(np.log(sim_times[1:]/3600), *popt))
ax_ch_len.annotate(
    f"Slope = {slope}",
    xy=(arr_head_x, arr_head_y),           # Point to annotate
    xytext=(2e4, 5),       # Text location
    arrowprops=dict(facecolor='red', arrowstyle='->', lw=2),
    fontsize=12,
    color='black',
    bbox=dict(facecolor='white', alpha=1, edgecolor='black')
)

## plot format
ax_ch_len.legend()
ax_ch_len.set_xscale('log')
ax_ch_len.set_yscale('log')
ax_ch_len.set_xlabel('t [s]')
ax_ch_len.set_ylabel('Characteristic length [nm]')
ax_ch_len.grid(True, which='both', linestyle='-', linewidth=0.5, alpha=0.7)
plot_file=os.path.join(plot_dir, 'ch_len.pdf')
fig_ch_len.savefig(plot_file, format='pdf')
plt.close(fig_ch_len)
