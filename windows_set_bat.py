"""
This script should be executed with the base conda environment activated.
A simple option is to open the Anaconda Navigator and execute this using
Spyder or Visual Studio Code. 
"""
import os

try:
    # find conda's activate (and select only the first one):
    activate = os.popen("where activate").read().split('\n')[0]

    # the full path to the gui file:
    progr = os.path.join(os.getcwd(), "gui_main.py")

    # the full command:
    full = f'@call "{activate}" pred-geomap & python "{progr}" "%~f0" %* & goto :eof'

    # save to file:
    with open("predictive_mapping.bat", "w") as fout:
        fout.write(full)
except:
    print("Oops! Something went wrong.")
else:
    print("Execution complete.")
