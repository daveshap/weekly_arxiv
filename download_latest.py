import requests
from time import time
from xml.etree import ElementTree as ET


# get search term
a = input("Whatcha wanna lookup? ")


# URL of the XML object
url = "https://export.arxiv.org/api/query?search_query=all:%s&sortBy=lastUpdatedDate&sortOrder=descending&max_results=200" % a.lower().replace(' ','%20')

# Send a GET request to the URL
response = requests.get(url)

# Parse the XML response
root = ET.fromstring(response.content)

# Namespace dictionary to find elements
namespaces = {'atom': 'http://www.w3.org/2005/Atom', 'arxiv': 'http://arxiv.org/schemas/atom'}

# Open the output file with UTF-8 encoding
with open("output-%s-%s.md" % (time(), a), "w", encoding='utf-8') as file:
    # Iterate over each entry in the XML data
    for entry in root.findall('atom:entry', namespaces):
        # Extract and write the title
        title = entry.find('atom:title', namespaces).text
        title = ' '.join(title.split())  # Replace newlines and superfluous whitespace with a single space
        file.write(f"# {title}\n\n")

        # Extract and write the link to the paper
        id = entry.find('atom:id', namespaces).text
        file.write(f"[Link to the paper]({id})\n\n")

        # Extract and write the authors
        authors = entry.findall('atom:author', namespaces)
        file.write("## Authors\n")
        for author in authors:
            name = author.find('atom:name', namespaces).text
            file.write(f"- {name}\n")
        file.write("\n")

        # Extract and write the summary
        summary = entry.find('atom:summary', namespaces).text
        file.write(f"## Summary\n{summary}\n\n")
