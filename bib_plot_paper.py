"""
This plots figure used in publication for ball in box model
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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import pyplot as plt
import numpy as np
import fitz

# ignore warnings
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

# analytical SAS function
def J1(x):
    if x==0:
        return 0
    else:
        return (np.sin(x)-x*np.cos(x))/x**2
    
def arrange_order(a,b,c):
    zusammen=[a,b,c]
    zusammen.sort()
    return zusammen[0], zusammen[1], zusammen[2]

def gauss_legendre_double_integrate(func, domain1, domain2, deg):
    x, w = np.polynomial.legendre.leggauss(deg)
    xgrid, ygrid=np.meshgrid(x,x)
    x=xgrid.reshape((1,np.size(xgrid)))
    y=ygrid.reshape((1,np.size(ygrid)))
    wx, wy=np.meshgrid(w,w)
    w1=wx.reshape((1,np.size(wx)))
    w2=wy.reshape((1,np.size(wy)))
    s1 = (domain1[1] - domain1[0])/2
    a1 = (domain1[1] + domain1[0])/2
    s2 = (domain2[1] - domain2[0])/2
    a2 = (domain2[1] + domain2[0])/2
    return np.sum(s1*s2*w1*w2*func(s1*x + a1,s2*y + a2))

def ball_in_box_ecc(qmax,qmin,Npts,scale,scale2,bg,sld_box, sld_ball,sld_sol,
                    length_a,length_b,length_c,radius, origin_shift):
    length_a, length_b, length_c=arrange_order(length_a,length_b,length_c)
    vol_box=length_a*length_b*length_c
    vol_ball=(4/3)*np.pi*radius**3
    # SLD unit 10^-5 \AA^-2
    del_rho_box=sld_box-sld_sol
    del_rho_ball=sld_ball-sld_sol
    q_arr=np.linspace(qmin,qmax,Npts) 
    Aq_arr=np.zeros(len(q_arr))
    for i in range(len(q_arr)):
        q=q_arr[i]
        if q==0:
            func=lambda alpha, psi:\
                np.abs(del_rho_box*vol_box*(np.sinc((1/np.pi)*q*length_a/2*np.sin(alpha)*np.sin(psi)))*\
                (np.sinc((1/np.pi)*q*length_b/2*np.sin(alpha)*np.cos(psi)))*\
                (np.sinc((1/np.pi)*q*length_c/2*np.cos(alpha))))**2*\
                    np.sin(alpha)
        else:
            func=lambda alpha, psi:\
                np.abs(del_rho_box*vol_box*(np.sinc((1/np.pi)*q*length_a/2*np.sin(alpha)*np.sin(psi)))*\
                (np.sinc((1/np.pi)*q*length_b/2*np.sin(alpha)*np.cos(psi)))*\
                (np.sinc((1/np.pi)*q*length_c/2*np.cos(alpha)))-\
                scale2*3*vol_ball*(del_rho_box-del_rho_ball)*\
                    J1(q*radius)/(q*radius)*\
                    np.vectorize(complex)(np.cos(q * np.sin(alpha) * np.sin(psi) * origin_shift[0]\
                                    + q * np.sin(alpha) * np.cos(psi) * origin_shift[1]\
                                        + q *  np.cos(psi) * origin_shift[2]),
                            np.sin(q * np.sin(alpha) * np.sin(psi) * origin_shift[0]\
                                    + q * np.sin(alpha) * np.cos(psi) * origin_shift[1]\
                                        + q *  np.cos(psi) * origin_shift[2])))**2*\
                    np.sin(alpha)
        
        psi_lim=np.pi
        alpha_lim=np.pi/2
        # Amplitude unit (\AA^3 * 10^-5 \AA^-2)^2 = 10^-10 \AA^2 
        Aq_arr[i]=(1/psi_lim)*gauss_legendre_double_integrate(func,[0, alpha_lim],[0, psi_lim],76)
    Iq_arr = scale*Aq_arr + bg # Intensity unit 10^-10 \AA^2    
    return Iq_arr, q_arr

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
sim_model='bib_ecc'
# simulation parameters
dt=1.
t_end=0.
n_ensem=1
# calculate time array
t_arr=np.arange(0,t_end+dt, dt)

### model_param ###
# model params
ball_rad=10
sld_in=2
sld_out=1
ecc_vec_arr=[np.array([5, 5, 5]), np.array([0, 0, 0])]
qclean_sld=0
# dir name for model param
# will be defined later

### scatt_cal ###
num_cat=3
method_cat='extend'
sig_file='signal.h5'
scan_vec=np.array([1, 0, 0])
Q_range=np.array([0., 1.])
num_points=100
num_orientation=300
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
# will be defined later

# create folder for figure (one level up from data folder)
figure_dir=os.path.join(mother_dir, '../../figure/')
os.makedirs(figure_dir, exist_ok=True)
## folder for this suit of figures
plot_dir=os.path.join(figure_dir, sim_model + '_vary')
os.makedirs(plot_dir, exist_ok=True)
# plott setting 
# plotting according to ecc vector
fig, ax = plt.subplots(figsize=(7, 5))
markers=['o', '^']
num_colors=['r', 'b']
ms_arr=[3, 3]
ana_linestyles=['-', '--']

for idx in range(len(ecc_vec_arr)):   
    # dir name for model param
    ecc_vec=ecc_vec_arr[idx]
    model_param_dir_name = ('rad' + '_' + str(ball_rad) + '_' +
                            'sld_in' + '_' + str(sld_in) + '_' +
                            'sld_out' + '_' + str(sld_out) + '_' +
                            'x' + '_' + str(ecc_vec[0]) + '_' + 
                            'y' + '_' + str(ecc_vec[1]) + '_' + 
                            'z' + '_' + str(ecc_vec[2]) + '_' + 
                            'qclean_sld' + '_' + str(qclean_sld)).replace('.', 'p')

    # folder name for model with particular run param
    model_param_dir=os.path.join(model_dir,model_param_dir_name)
    for i in range(len(t_arr)):
        t=t_arr[i]
        # time_dir name
        t_dir_name='t{0:0>3}'.format(i)
        t_dir=os.path.join(model_param_dir, t_dir_name)

        # box geometry
        vol_box=length_a*length_b*length_c

        # volume for normalization
        vol_norm=vol_box

        # numerical intensity
        ## read I vs Q signal file
        Iq_data_file_name='Iq_{0}.h5'.format(scatt_settings) 
        Iq_data_file=os.path.join(t_dir,Iq_data_file_name)
        Iq_data=h5py.File(Iq_data_file,'r')
        Iq=Iq_data['Iq'][:] # unit fm^2
        Iq=Iq/vol_norm*10**2 # unit (fm^2 / \AA^3) * 10^2 = cm^-1
        q=Iq_data['Q'][:]
        Iq_data.close()

        # ananlytical intensity 
        ## Intensity unit 10^-10 \AA^2
        Iq_ana,q_ana= ball_in_box_ecc(qmax=np.max(q),qmin=np.min(q),Npts=100,
                    scale=1,scale2=1,bg=0,sld_box=sld_out, sld_ball=sld_in, sld_sol=0,
                    length_a=length_a, length_b=length_b, length_c=length_c, radius=ball_rad, 
                    origin_shift=ecc_vec)
        
        ## Normalize by volume
        ## (Before * 10**2) Intensity unit 10^-10 \AA^-1 = 10 ^-2 cm^-1
        ## (after * 10**2) Intensity unit cm^-1
        Iq_ana = (Iq_ana / vol_norm) * 10**2
        
        # loglog plot
        ax.loglog(q, Iq,color=num_colors[idx], linestyle='', marker=markers[idx], markersize=ms_arr[idx], 
                  label= r'Numerical calculation: $ \overrightarrow{R}_{\mathrm{ecc}} = $' + '({0}, {1}, {2})'.format(ecc_vec[0], ecc_vec[1], ecc_vec[2]))
        ax.loglog(q_ana, Iq_ana, 'gray', linestyle=ana_linestyles[idx], 
                  label= r'Analytical calculation: $ \overrightarrow{R}_{\mathrm{ecc}} = $' + '({0}, {1}, {2})'.format(ecc_vec[0], ecc_vec[1], ecc_vec[2]))

        # plot formatting
        ## legend
        ax.legend(loc='lower left')
        ## labels
        ax.set_xlabel('Q [$\mathrm{\AA}^{-1}$]')
        ax.set_ylabel('I(Q) [$\mathrm{cm}^{-1}$]')
        ## SANS upper boundary Q=1 \AA^-1
        ax.set_xlim(right=1)

ax_ins1 = inset_axes(ax, width="100%", height="100%", bbox_to_anchor=(0, 0.4, 0.5, 0.5),
        bbox_transform=ax.transAxes)

ax_ins1.axis('off')

# Load the PDF and select the first page
doc = fitz.open('./ppt/ballinboxEccentric.pdf')
page = doc.load_page(0)

# Define crop area (x0, y0, x1, y1) in points
crop_rect = fitz.Rect(250, 50, 550, 350)  # adjust as needed

# Render the cropped area to a pixmap (image)
pix = page.get_pixmap(clip=crop_rect, dpi=200)
img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

# Plot with matplotlib
ax_ins1.imshow(img)

# save plot
plot_file_name='Iq_bib_ecc.pdf'
plot_file=os.path.join(plot_dir,plot_file_name)
plt.savefig(plot_file, format='pdf')
plt.close(fig)