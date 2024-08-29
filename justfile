#    <one line to give the program's name and a brief idea of what it does.>  
#    Copyright © 2024 Charudatta
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
#    email contact: 152109007c@gmailcom

set windows-shell := ["pwsh.exe", "-NoLogo", "-Command"]

default:
    @just --choose

# create files and directories
init:
    #!pwsh
    git init
    New-Item -ItemType "file" -Path ".gitattribute", "main.py", "requirements.json", "config.json"
    New-Item -ItemType "directory" -Path "docs", "src", "tests"
    New-Item -ItemType "file" -Path .\* -Name "__init__.py" -ErrorAction SilentlyContinue
    gig gen python > .gitignore 
    Add-LicenseHeader
    7z a archives.7z .gitignore

# set configuration variables
config:
    #!pwsh
    config.json >> .gitignore
    Set-EnvFromJson

# add documentation to repo
docs:
    #!pwsh
    conda activate blog
    python -m mkdocs new .

# genearte and readme to repo    
readme:
    #!pwsh
    conda activate w
    python C:/Users/$env:username/Documents/GitHub/readmeGen/main.py

# version control repo with git
commit message="init":
    #!pwsh
    git add .
    git commit -m {{message}}

# create windows executable
exe file_name:
    #!pwsh
    pyinstaller src/{{file_name}} --onefile

# run python unit test 
tests:
    #!pwsh
    python -m unittest discover -s tests

# Add custom tasks, enviroment variables

cgan:
    #!pwsh
    cd src
    conda activate w
    p RunGAN.py
    


