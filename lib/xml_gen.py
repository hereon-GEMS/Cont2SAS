import xml.etree.ElementTree as ET
import os

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
