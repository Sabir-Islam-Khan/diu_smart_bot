import xml.etree.ElementTree as ET

def extract_urls_from_xml(xml_file):
    # Parse the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Define a list to store the URLs
    urls = []

    # Iterate through all 'url' elements in the XML tree
    for url_elem in root.findall('{http://www.sitemaps.org/schemas/sitemap/0.9}url'):
        # Find the 'loc' element inside each 'url' element
        loc_elem = url_elem.find('{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
        if loc_elem is not None:
            urls.append(loc_elem.text)

    return urls

def save_urls_to_file(urls, output_file):
    with open(output_file, 'w') as file:
        file.write("[\n")
        for url in urls:
            file.write(f'    "{url}",\n')
        file.write("]\n")

# Example usage:
if __name__ == "__main__":
    xml_file = "sitemap.xml"  # Replace with the path to your XML file
    output_file = "urls.txt"  # Output file path
    urls = extract_urls_from_xml(xml_file)
    save_urls_to_file(urls, output_file)
    print(f"Extracted URLs saved to '{output_file}'.")
