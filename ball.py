from lib import xml_gen
import subprocess

"""
input values
"""
### struct gen ###
xml_dir='./xml' 
length_a=40 
length_b=40 
length_c=40
nx=40 
ny=40 
nz=40 
el_type='lagrangian'
el_order=1
update_val=True
plt_node=False
plt_cell=False
plt_mesh=False

### sim_gen ###
sim_model='ball'
dt=1 
t_end=0
n_ensem=1

### model_param ###
rad=15
sld=2


"""
input values
"""
# generate xml for struct_gen
xml_gen.struct_xml_write(xml_dir, 
                         length_a, length_b, length_c, 
                         nx, ny, nz, 
                         el_type, el_order,  
                         update_val, plt_node, plt_cell, plt_mesh)

# generate structure
script_path = "src/struct_gen.py"  # Full path of the script
working_dir = "."  # Directory to run the script in
subprocess.run(["python", script_path], cwd=working_dir)

# simulation xml
xml_gen.sim_xml_write(xml_dir, sim_model,dt, t_end, n_ensem)

# model xml
xml_gen.model_ball_xml_write(xml_dir, rad, sld)

# generate structure
script_path = "src/sim_gen.py"  # Full path of the script
working_dir = "."  # Directory to run the script in
subprocess.run(["python", script_path], cwd=working_dir)
