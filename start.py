# Copyright 2024 Charudatta
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import subprocess

if input("Do you want create directories? ")== "y":
    list_dirs = ["archive", "data", "src", "test"]
    for dir in list_dirs:
        if not os.path.exists(dir):
            print('Creating Directory: ' + dir)
            os.makedirs(dir)
            with open(os.path.join(dir, "__init__.py"), "w") as f:
                pass

        
if input("Do you want to create files? ")== "y":
    list_files = ["license", "__main__.py", "requirements.txt", ".gitattributes"]
    for file in list_files:
        if not os.path.exists(file):
            print('Creating Files: ' + file)
            with open(file, "w") as f:
                pass  

if input("Is repo version controlled with git? ")== "y":
    subprocess.run(["git", "init"])
    subprocess.run(["gig", "gen", "python", ">", ".gitignore"])
    
if input("Do you want to add License header? ")== "y":   
    subprocess.run(["nwa", "add", "gpl3", ".", "-c", "Charudatta"]) 

if input("Is repo is having mkdocs documentation? ")== "y": 
    subprocess.run(["mkdocs", "new", "."])

if input("Is repo configuration managed by dynaconf? ")== "y": 
    subprocess.run(["dynaconf", "init","-f","json"])

if input("Is readme and readmex needed to be generated? ")== "y":
    subprocess.run(["python", "C:/Users/chaitrali/Documents/GitHub/readme-generator"])
    
if input("Do you want to create assets directories? ")== "y":
    list_dirs = ["docs/assets","docs/assets/css", "docs/assets/img", "docs/assets/js"]
    for dir in list_dirs:
        if not os.path.exists(dir):
            print('Creating Directory: ' + dir)
            os.mkdir(dir) 

if input("Do you want to create initial commit? ")== "y":
    subprocess.run(["git", "add","."])
    subprocess.run(["git", "commit","-m","Initial commit"])
        
if input("Do you want create executable file?")== "y":
    subprocess.run(["pyintsaller", f"__main__.py", "-onefile"])
