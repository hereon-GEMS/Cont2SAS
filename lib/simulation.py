
import numpy as np
from scipy import special
from scipy.optimize import minimize
#from numba import njit, prange
from scipy.special import erf
import xml.etree.ElementTree as ET
import os

def model_run(sim_model,nodes, midpoint, t, t_end):
    if sim_model=='ball':
        return model_ball(nodes, midpoint, t)
    if sim_model=='box':
        return model_box(nodes, midpoint, t)
    if sim_model=='bib':
        return model_bib(nodes, midpoint, t)
    if sim_model=='bib_ecc':
        return model_bib_ecc(nodes, midpoint, t)
    if sim_model=='gg':
        return model_gg(nodes, midpoint, t, t_end)
    if sim_model=='fs':
        return model_fs(nodes, midpoint, t, t_end)

def model_ball(nodes, midpoint, t):
    # read model_run_param from xml
    xml_folder='../xml/'
    struct_xml=os.path.join(xml_folder, 'model_ball.xml')
    tree=ET.parse(struct_xml)
    root = tree.getroot()
    # read params
    rad=float(root.find('rad').text)
    sld=float(root.find('sld').text)
    # run simulation
    nodes = np.array(nodes)
    sim_sld = np.zeros(len(nodes))
    cord_ed = np.sum((nodes-midpoint)**2,axis=1)
    sim_sld [cord_ed <= rad**2] = sld
    sld_max=sld
    sld_min=0
    return sim_sld, sld_max, sld_min
    return sim_sld

def model_box(nodes, midpoint, t):
    # read model_run_param from xml
    xml_folder='../xml/'
    struct_xml=os.path.join(xml_folder, 'model_box.xml')
    tree=ET.parse(struct_xml)
    root = tree.getroot()
    # read params
    sld=float(root.find('sld').text)
    # run simulation
    nodes = np.array(nodes)
    sim_sld = sld*np.ones(len(nodes))
    sld_max=sld
    sld_min=0
    return sim_sld, sld_max, sld_min

def model_bib(nodes, midpoint, t):
    # read model_run_param from xml
    xml_folder='../xml/'
    struct_xml=os.path.join(xml_folder, 'model_bib.xml')
    tree=ET.parse(struct_xml)
    root = tree.getroot()
    # read params
    rad=float(root.find('rad').text)
    sld_ball=float(root.find('sld_in').text)
    sld_box=float(root.find('sld_out').text)
    # run simulation
    nodes = np.array(nodes)
    sim_sld = sld_box*np.ones(len(nodes))
    cord_ed = np.sum((nodes-midpoint)**2,axis=1)
    sim_sld [cord_ed <= rad**2] = sld_ball
    sld_max=sld_ball
    sld_min=sld_box
    return sim_sld, sld_max, sld_min

def model_bib_ecc(nodes, midpoint, t):
    # read model_run_param from xml
    xml_folder='../xml/'
    struct_xml=os.path.join(xml_folder, 'model_bib_ecc.xml')
    tree=ET.parse(struct_xml)
    root = tree.getroot()
    # read params
    rad=float(root.find('rad').text)
    sld_ball=float(root.find('sld_in').text)
    sld_box=float(root.find('sld_out').text)
    ecc_x=float(root.find('ecc').find('x').text)
    ecc_y=float(root.find('ecc').find('y').text)
    ecc_z=float(root.find('ecc').find('z').text)
    ecc=np.array([ecc_x, ecc_y, ecc_z])
    # run simulation
    nodes = np.array(nodes)
    sim_sld = sld_box*np.ones(len(nodes))
    ball_mid=midpoint+ecc
    cord_ed = np.sum((nodes-ball_mid)**2,axis=1)
    sim_sld [cord_ed <= rad**2] = sld_ball
    sld_max=sld_ball
    sld_min=sld_box
    return sim_sld, sld_max, sld_min

def model_gg(nodes, midpoint, t, t_end):
    # read model_run_param from xml
    xml_folder='../xml/'
    struct_xml=os.path.join(xml_folder, 'model_gg.xml')
    tree=ET.parse(struct_xml)
    root = tree.getroot()
    # read params
    rad_0=float(root.find('rad_0').text)
    rad_end=float(root.find('rad_end').text)
    sld_grain=float(root.find('sld_in').text)
    sld_env=float(root.find('sld_out').text)
    # run simulation
    nodes = np.array(nodes)
    sim_sld = sld_env*np.ones(len(nodes))
    rad=rad_0+t*(rad_end-rad_0)/t_end
    cord_ed = np.sum((nodes-midpoint)**2,axis=1)
    sim_sld [cord_ed <= rad**2] = sld_grain
    sld_max=sld_grain
    sld_min=sld_env
    return sim_sld, sld_max, sld_min

def model_fs(nodes, midpoint, t, t_end):
    # read model_run_param from xml
    xml_folder='../xml/'
    struct_xml=os.path.join(xml_folder, 'model_fs.xml')
    tree=ET.parse(struct_xml)
    root = tree.getroot()
    # read params
    rad=float(root.find('rad').text)
    sig_0=float(root.find('sig_0').text)
    sig_end=float(root.find('sig_end').text)
    sld_grain=float(root.find('sld_in').text)
    sld_env=float(root.find('sld_out').text)
    # run simulation
    nodes = np.array(nodes)
    sig=sig_0+t*(sig_end-sig_0)/t_end
    sim_sld = np.ones(len(nodes))
    coord_r=np.sqrt(np.sum((nodes-midpoint)**2,axis=1))
    if sig==0:
        sim_sld=(sld_env-sld_grain)*np.heaviside(coord_r-rad,0)+sld_grain
    else:
        #sld=-np.heaviside(coord_r-r,0)+1
        sim_sld=((sld_grain-sld_env)/2)*(1-special.erf((coord_r-rad)/(np.sqrt(2)*sig)))+sld_env
    sld_max=sld_grain
    sld_min=sld_env
    return sim_sld, sld_max, sld_min

########################## old functions #####################
### 2d function ###

### static grain models ###
# spherical grain

def sph_grain(coords,origin,r,sld_in,sld_out):
    coords = np.array(coords)
    sld = sld_out*np.ones(len(coords))
    cord_ed = np.sum((coords-origin)**2,axis=1)
    sld [cord_ed <= r**2] = sld_in
    return sld

### 3d function ###

### static grain models ###
# spherical grain


def sph_grain_3d(coords,origin,r,sld_in,sld_out):
    coords = np.array(coords)
    sld = sld_out*np.ones(len(coords))
    cord_ed = np.sum((coords-origin)**2,axis=1)
    sld [cord_ed <= r**2] = sld_in
    return sld

### dyanmic grain models ###
# spherical grain

# diffusion into the grain
# def sph_grain_diffus_3d(coords,origin,r,fuzz_val,sld_in,sld_out):
#     #r_arr=np.linspace(r,0,len(t))
#     coords = np.array(coords)
#     #sld = sld_out*np.ones((len(coords),len(t)))
#     sld = np.zeros(len(coords))
#     #cord_ed = np.sum((coords-origin)**2,axis=1)
#     coord_r=np.sqrt(np.sum((coords-origin)**2,axis=1))
#     if fuzz_val==0:
#         sld=-np.heaviside(coord_r-r,0)+1
#     else:
#         #sld=-np.heaviside(coord_r-r,0)+1
#         sld=(1/2)*(1-special.erf((coord_r-r)/(np.sqrt(2)*fuzz_val)))
#     return sld

def sph_grain_fs_3d(coords,origin,r,fuzz_val,sld_in,sld_out):
    #r_arr=np.linspace(r,0,len(t))
    coords = np.array(coords)
    #sld = sld_out*np.ones((len(coords),len(t)))
    sld = np.zeros(len(coords))
    #cord_ed = np.sum((coords-origin)**2,axis=1)
    coord_r=np.sqrt(np.sum((coords-origin)**2,axis=1))
    max_sld=np.max(sld_in, sld_out)
    min_sld=np.min(sld_in, sld_out)
    if fuzz_val==0:
        sld=-np.heaviside(coord_r-r,0)+1
    else:
        #sld=-np.heaviside(coord_r-r,0)+1
        sld=(1/2)*(1-special.erf((coord_r-r)/(np.sqrt(2)*fuzz_val)))
    #adjust with range
    sld_final = (sld_in-sld_out) * sld + sld_out
    #sld_final = sld
    return sld_final

def sph_grain_diffus_book_1_3d(nodes,origin,rad,D_coeff, time, sld_in,sld_out):
    n=50 # num_term in series
    a=rad
    t=time
    c1=sld_in
    c0=sld_out
    nodes = np.array(nodes)
    sld = sld_out*np.ones(len(nodes))
    r=np.sqrt(np.sum((nodes-origin)**2,axis=1))
    if t==0:
        sld [r**2 <= rad**2] = sld_in
            
    else:
        if D_coeff*t<=0.1:
            for i in range(len(r)):
                series=0
                if r[i] <=rad:
                    if r[i]==0:
                        for j in range(1,n):
                            series+=((-1)**j) * np.exp(-((D_coeff*(j**2)*(np.pi**2)*t)/(a**2)))
                    
                        sld[i]=c1+(c0-c1)*(1+2*series)
                    else:
                        for j in range(n):
                            #series+=((-1)**j/j) * np.sin(j*np.pi*r[i]/a) * np.exp(-((D_coeff*(j**2)*(np.pi**2)*t)/(a**2)))
                            term_1=(2*j+1)*a
                            term_2=2* np.sqrt(D_coeff*t)
                            series+=erf((term_1-r[i])/term_2)-erf((term_1+r[i])/term_2)
                        sld[i]=c1+(c0-c1)*(a/r[i]*series)
        else:
            for i in range(len(r)):
                series=0
                if r[i] <=rad:
                    if r[i]==0:
                        for j in range(1,n):
                            series+=((-1)**j) * np.exp(-((D_coeff*(j**2)*(np.pi**2)*t)/(a**2)))
                    
                        sld[i]=c1+(c0-c1)*(1+2*series)
                    else:
                        for j in range(1,n):
                            series+=((-1)**j/j) * np.sin(j*np.pi*r[i]/a) * np.exp(-((D_coeff*(j**2)*(np.pi**2)*t)/(a**2)))
                    
                        sld[i]=c1+(c0-c1)*(1+((2*a)/(np.pi*r[i]))*series)
    return sld
    

def sph_grain_hydinout_3d(nodes,origin,rad,D_coeff, time, sld_hyd, sld_dehyd,sld_out, cur_cond):
    n=50 # num_term in series
    a=rad
    t=time
    if cur_cond=='hyd':
        c1= sld_hyd#sld_in
        c0= sld_dehyd#sld_out
    else:
        c1= sld_dehyd#sld_in
        c0= sld_hyd#sld_out
    
    nodes = np.array(nodes)
    sld = sld_out*np.ones(len(nodes))
    r=np.sqrt(np.sum((nodes-origin)**2,axis=1))
    if t==0:
        sld [r**2 <= rad**2] = c1
            
    else:
        if D_coeff*t<=0.1:
            for i in range(len(r)):
                series=0
                if r[i] <=rad:
                    if r[i]==0:
                        for j in range(1,n):
                            series+=((-1)**j) * np.exp(-((D_coeff*(j**2)*(np.pi**2)*t)/(a**2)))
                    
                        sld[i]=c1+(c0-c1)*(1+2*series)
                    else:
                        for j in range(n):
                            #series+=((-1)**j/j) * np.sin(j*np.pi*r[i]/a) * np.exp(-((D_coeff*(j**2)*(np.pi**2)*t)/(a**2)))
                            term_1=(2*j+1)*a
                            term_2=2* np.sqrt(D_coeff*t)
                            series+=erf((term_1-r[i])/term_2)-erf((term_1+r[i])/term_2)
                        sld[i]=c1+(c0-c1)*(a/r[i]*series)
        else:
            for i in range(len(r)):
                series=0
                if r[i] <=rad:
                    if r[i]==0:
                        for j in range(1,n):
                            series+=((-1)**j) * np.exp(-((D_coeff*(j**2)*(np.pi**2)*t)/(a**2)))
                    
                        sld[i]=c1+(c0-c1)*(1+2*series)
                    else:
                        for j in range(1,n):
                            series+=((-1)**j/j) * np.sin(j*np.pi*r[i]/a) * np.exp(-((D_coeff*(j**2)*(np.pi**2)*t)/(a**2)))
                    
                        sld[i]=c1+(c0-c1)*(1+((2*a)/(np.pi*r[i]))*series)
    return sld
    
def sph_multigrain_loc_3d(coords,r_val,r_num,sld_in,sld_out, box_length):
    #print('we are in')
    
    # Set up the initial configuration (positions and radii)
    #rad_val=np.linspace(1,5,5)
    #rad_num=np.array([2, 4, 3, 2, 5])
    # rad_val=np.linspace(1,5,2)
    # rad_num=np.array([2, 2])
    rad_val=r_val
    rad_num=r_num#np.array(r_num, dtype='int')
    radii=np.zeros(int(np.sum(rad_num)))
    idx=0
    for i in range(len(rad_val)):
        num=int(rad_num[i])
        radii[idx:idx+num]=np.repeat(rad_val[i], num)
        idx+=num
    
    box_length_x=box_length
    box_length_y=box_length_x
    box_length_z=box_length_x
    initial_positions=np.row_stack((box_length_x*np.random.random(len(radii)), 
                             box_length_y*np.random.random(len(radii)), 
                             box_length_z*np.random.random(len(radii)))).T 
    
    # Define the Lennard-Jones potential function
    def lj_potential(positions, radii=radii):
        potential_energy = 0.0
        tot_el=len(positions)
        positions=positions.reshape(int(tot_el/3),3)
        num_spheres = len(positions)
    
        for i in range(num_spheres):
            for j in range(i + 1, num_spheres):
                #print(i,j)
                rij = np.linalg.norm(positions[i] - positions[j])
                sigma = (radii[i] + radii[j]) / 2.0
                epsilon = 1.0  # Adjust as needed
                potential_energy += 4.0 * epsilon * ((sigma / rij) ** 12 - (sigma / rij) ** 6)
                for k in range(3):
                    if positions[i,k]-radii[i] <=0:
                        potential_energy +=1e9
    
        return potential_energy
    
    # Define constraints to prevent overlap
    def constraint_overlap(positions, radii=radii):
        tot_el=len(positions)
        positions=positions.reshape(int(tot_el/3),3)
        num_spheres = len(positions)
        constraints = []
    
        for i in range(num_spheres):
            for j in range(i + 1, num_spheres):
                rij = np.linalg.norm(positions[i] - positions[j])
                sigma = (radii[i] + radii[j]) / 2.0
                #constraints.append(sigma - rij)
                constraints.append(-sigma + rij) 
        
        return constraints
    
    # Perform energy minimization with constraints
    result = minimize(
        lj_potential,
        x0=initial_positions,
        args=(radii,),
        constraints={"type": "ineq", "fun": constraint_overlap, 'args': (radii,)},
        method="SLSQP"  # You can choose a different optimization method if needed
    )
    
    for i in range(len(radii)):
        print('coordinate: {0}, radius: {1}'.format(result.x.reshape(len(radii),3)[i], radii[i]))
    
    return result.x.reshape(len(radii),3), radii

#@njit(parallel=True)    
def sph_multigrain_3d(nodes,radii, r_dist, sld_in,sld_out):
    print('we are in')
    nodes = np.array(nodes)
    sld = sld_out*np.ones(len(nodes))
    for i in range(len(r_dist)):
        origin=r_dist[i]
        r=radii[i]
        #print(origin)
        cord_ed = np.sum((nodes-origin)**2,axis=1)
        sld [cord_ed <= r**2] = sld_in
    return sld
    
    