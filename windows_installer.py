"""
This script should be executed with the base conda environment activated.
A simple option is to open the Anaconda Navigator and execute this using
Spyder or Visual Studio Code. 
"""

import os

# create the environment:
os.system("conda env create -f environment.yml")

# create .bat file:
os.system("python windows_set_bat.py")

print("Execution complete.")