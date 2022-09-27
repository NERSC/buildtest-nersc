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
    print("Provide a valid template file. "+file_path+" not found.")
    exit()

if not os.path.isfile(package_path):
    print("Provide a valid package-list.yml file. "+package_path+" not found.")
    exit()

packages=[]
with open(package_path, 'r') as stream:
    packages = yaml.safe_load(stream)

text_file = open(file_path, "r")
 
template = text_file.read()
text_file.close()

for pname in packages["package"]:
    ymlOut = open(pname+".yml", "w")
    ymlOut.write(template.format(package=pname))
    ymlOut.close()
