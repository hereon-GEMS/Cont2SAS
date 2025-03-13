import xml.etree.ElementTree as ET
from xml.dom.minidom import parseString
import os

# write struct_gen.xml file
def struct_xml_write(xml_dir, length_a, length_b, length_c, nx, ny, nz, el_type, el_order,
                     update_val=True, plt_node=False, plt_cell=False, plt_mesh=False):
    # Create the root element
    root = ET.Element("root")

    # details for simulation box
    lengths = ET.SubElement(root, "lengths")
    ET.SubElement(lengths, "x").text=str(length_a)
    ET.SubElement(lengths, "y").text=str(length_b)
    ET.SubElement(lengths, "z").text=str(length_c)

    # number of cells in each direction
    num_cell = ET.SubElement(root, "num_cell")
    ET.SubElement(num_cell, "x").text=str(nx)
    ET.SubElement(num_cell, "y").text=str(ny)
    ET.SubElement(num_cell, "z").text=str(nz)

    # element details (lagrangian 1 and 2 possible)
    element = ET.SubElement(root, "element")
    ET.SubElement(element, "type").text=str(el_type)
    ET.SubElement(element, "order").text=str(el_order)
    
    # decision to update plot or not
    decision = ET.SubElement(root, "decision")
    ET.SubElement(decision, "update").text=str(update_val)
    plot = ET.SubElement(decision, "plot")
    ET.SubElement(plot, "node").text=str(plt_node)
    ET.SubElement(plot, "cell").text=str(plt_cell)
    ET.SubElement(plot, "mesh").text=str(plt_mesh)

    # Convert to a string and format
    tree = ET.ElementTree(root)
    xml_str = ET.tostring(root, encoding="utf-8", method="xml").decode()
    formatted_xml = parseString(xml_str).toprettyxml(indent="  ")

    # Save to a file
    xml_file_name='struct.xml'
    xml_file=os.path.join(xml_dir, xml_file_name)
    with open(xml_file, "w", encoding="utf-8") as f:
        f.write(formatted_xml)
    
    print("structure generation xml file created successfully!")

# write simulation.xml file
def sim_xml_write(xml_dir, sim_model,dt, t_end, n_ensem):
    # Create the root element
    root = ET.Element("root")

    # details for simulation box
    ET.SubElement(root, "model").text=str(sim_model)
    sim_param=ET.SubElement(root, "sim_param")
    ET.SubElement(sim_param, "dt").text=str(dt)
    ET.SubElement(sim_param, "tend").text=str(t_end)
    ET.SubElement(sim_param, "n_ensem").text=str(n_ensem)

    # Convert to a string and format
    tree = ET.ElementTree(root)
    xml_str = ET.tostring(root, encoding="utf-8", method="xml").decode()

    # Save to a file
    xml_file_name='simulation.xml'
    xml_file=os.path.join(xml_dir, xml_file_name)
    formatted_xml = parseString(xml_str).toprettyxml(indent="  ")

    with open(xml_file, "w", encoding="utf-8") as f:
        f.write(formatted_xml)
    
    print("simulation xml file created successfully!")

# write model_ball.xml file
def model_ball_xml_write(xml_dir, rad, sld):
    # Create the root element
    root = ET.Element("root")

    # details for simulation box
    ET.SubElement(root, "rad").text=str(rad)
    ET.SubElement(root, "sld").text=str(sld)

    # Convert to a string and format
    tree = ET.ElementTree(root)
    xml_str = ET.tostring(root, encoding="utf-8", method="xml").decode()

    # Save to a file
    xml_file_name='model_ball.xml'
    xml_file=os.path.join(xml_dir, xml_file_name)
    formatted_xml = parseString(xml_str).toprettyxml(indent="  ")

    with open(xml_file, "w", encoding="utf-8") as f:
        f.write(formatted_xml)
    
    print("model_ball xml file created successfully!")

# write model_box.xml file
def model_box_xml_write(xml_dir, sld):
    # Create the root element
    root = ET.Element("root")

    # details for simulation box
    ET.SubElement(root, "sld").text=str(sld)

    # Convert to a string and format
    tree = ET.ElementTree(root)
    xml_str = ET.tostring(root, encoding="utf-8", method="xml").decode()

    # Save to a file
    xml_file_name='model_box.xml'
    xml_file=os.path.join(xml_dir, xml_file_name)
    formatted_xml = parseString(xml_str).toprettyxml(indent="  ")

    with open(xml_file, "w", encoding="utf-8") as f:
        f.write(formatted_xml)
    
    print("model_box xml file created successfully!")

# write model_fs.xml file
def model_fs_xml_write(xml_dir, rad, sig_0, sig_end, sld_in, sld_out):
    # Create the root element
    root = ET.Element("root")

    # details for simulation box
    ET.SubElement(root, "rad").text=str(rad)
    ET.SubElement(root, "sig_0").text=str(sig_0)
    ET.SubElement(root, "sig_end").text=str(sig_end)
    ET.SubElement(root, "sld_in").text=str(sld_in)
    ET.SubElement(root, "sld_out").text=str(sld_out)

    # Convert to a string and format
    tree = ET.ElementTree(root)
    xml_str = ET.tostring(root, encoding="utf-8", method="xml").decode()

    # Save to a file
    xml_file_name='model_fs.xml'
    xml_file=os.path.join(xml_dir, xml_file_name)
    formatted_xml = parseString(xml_str).toprettyxml(indent="  ")

    with open(xml_file, "w", encoding="utf-8") as f:
        f.write(formatted_xml)
    
    print("model_fs xml file created successfully!")

# write scatt_cal.xml file
def scatt_cal_xml_write(xml_dir, num_cat, method_cat, 
                        sassena_exe, mpi_procs, num_threads, 
                        sig_file, scan_vec_val, Q_range,
                        num_points, num_orientation):
    
    # Create the root element
    root = ET.Element("root")

    # details for categorization
    discretization = ET.SubElement(root, "discretization")
    ET.SubElement(discretization, "num_cat").text=str(num_cat)
    ET.SubElement(discretization, "method_cat").text=str(method_cat)

    # details for sassena
    sassena = ET.SubElement(root, "sassena")
    ET.SubElement(sassena, "exe_loc").text=str(sassena_exe)
    ET.SubElement(sassena, "mpi_procs").text=str(mpi_procs)
    ET.SubElement(sassena, "num_threads").text=str(num_threads)

    # details for scatt_cal
    scatt_cal = ET.SubElement(root, "scatt_cal")
    ET.SubElement(scatt_cal, "sig_file").text=str(sig_file)
    scan_vec=ET.SubElement(scatt_cal, "scan_vec")
    ET.SubElement(scan_vec, "x").text=str(scan_vec_val[0])
    ET.SubElement(scan_vec, "y").text=str(scan_vec_val[1])
    ET.SubElement(scan_vec, "z").text=str(scan_vec_val[2])
    ET.SubElement(scatt_cal, "Q_start").text=str(Q_range[0])
    ET.SubElement(scatt_cal, "Q_end").text=str(Q_range[1])
    ET.SubElement(scatt_cal, "num_points").text=str(num_points)
    ET.SubElement(scatt_cal, "num_orientation").text=str(num_orientation)

    # Convert to a string and format
    tree = ET.ElementTree(root)
    xml_str = ET.tostring(root, encoding="utf-8", method="xml").decode()

    # Save to a file
    xml_file_name='scatt_cal.xml'
    xml_file=os.path.join(xml_dir, xml_file_name)
    formatted_xml = parseString(xml_str).toprettyxml(indent="  ")

    with open(xml_file, "w", encoding="utf-8") as f:
        f.write(formatted_xml)
    
    print("scatt_cal xml file created successfully!")

