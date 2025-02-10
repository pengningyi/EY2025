import xml.etree.ElementTree as ET
"""
The output you see is the parsed data from the KML file. 
Each section represents a Placemark element, and the script extracts 
and prints the following information for each Placemark:

Coordinates: The geographical coordinates of the Placemark.
ExtendedData: Additional data associated with the Placemark, including:
tessellate: Indicates whether to allow the geometry to follow the terrain.
extrude: Indicates whether to extend the geometry to the ground.
visibility: Indicates whether the Placemark is visible.
id: A unique identifier for the Placemark.
fid: Another identifier, possibly a feature ID.
layer: The layer name associated with the Placemark.
path: The file path and layer name associated with the Placemark.
Each Placemark is separated by --- in the output. 


This indicates that the Placemark has the specified coordinates and extended data attributes. 
The coordinates represent the vertices of the polygon that defines the footprint of a building.
The extended data provides additional metadata about the Placemark.
"""
#input your kml file path
kml_file_path = r"\Building_Footprint.kml"

# Parse the KML file
tree = ET.parse(kml_file_path)
root = tree.getroot()

# Define the namespace
namespace = {'kml': 'http://www.opengis.net/kml/2.2'}

# Iterate through each Placemark element
for placemark in root.findall('.//kml:Placemark', namespace):
    # Extract and print the name of the Placemark
    name = placemark.find('kml:name', namespace)
    if name is not None:
        print(f'Name: {name.text}')
    
    # Extract and print the coordinates of the Placemark
    coordinates = placemark.find('.//kml:coordinates', namespace)
    if coordinates is not None:
        print(f'Coordinates: {coordinates.text}')
    
    # Extract and print the ExtendedData of the Placemark
    extended_data = placemark.find('kml:ExtendedData', namespace)
    if extended_data is not None:
        for data in extended_data.findall('.//kml:SimpleData', namespace):
            print(f'{data.attrib["name"]}: {data.text}')
    
    print('---')

