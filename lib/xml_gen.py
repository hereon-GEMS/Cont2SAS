import xml.etree.ElementTree as ET
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

    # Save to a file
    xml_file_name='struct_gen.xml'
    xml_file=os.path.join(xml_dir, xml_file_name)
    with open(xml_file, "w", encoding="utf-8") as f:
        f.write(xml_str)
    
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
    with open(xml_file, "w", encoding="utf-8") as f:
        f.write(xml_str)
    
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
    with open(xml_file, "w", encoding="utf-8") as f:
        f.write(xml_str)
    
    print("model_ball xml file created successfully!")