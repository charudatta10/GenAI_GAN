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
#    


if ('y' -eq (Read-Host "Is repo git version controlloed? y/n ")) 
{
    git init

}


if ('y' -eq (Read-Host "Do you want to customize readme? y/n ")) 
{
    python C:/Users/chaitrali/Documents/GitHub/readme-generator

}    

if ('y' -eq (Read-Host "Do you want to create initial commit? y/n ")) 
{
    git add .
    git commit -m "initial commit"

}


        
 









