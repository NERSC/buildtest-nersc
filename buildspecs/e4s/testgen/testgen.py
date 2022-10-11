#!/usr/bin/env python3

import os
import sys
import yaml

if len(sys.argv) < 3:
    print("Provide a template file followed by a package-list.yml file.")
    exit()
 
file_path = sys.argv[1]
package_path = sys.argv[2]
 
#check if file is present
if not os.path.isfile(file_path):
    exit("ERROR: Provide a valid template file. "+file_path+" not found.")

if not os.path.isfile(package_path):
    exit("ERROR: Provide a valid package-list.yml file. "+package_path+" not found.")

packages=[]
with open(package_path, 'r') as stream:
    packages = yaml.safe_load(stream)

if packages is None or "package" not in packages:
    exit("Error: No package entry found in "+package_path)

template=""
with  open(file_path, "r") as text_file:
    template = text_file.read()

if template is None or not template.strip():
    exit("ERROR: Empty template file. "+file_path)

for pname in packages["package"]:
    with open(pname+".yml", "w") as ymlOut:
        ymlOut.write(template.format(package=pname))
