import os

try:
    # find conda's activate (and strip breakline):
    activate = os.popen("where activate").read().rstrip('\n')

    # the full path to the gui file:
    progr = os.path.join(os.getcwd(), "gui_main.py")

    # the full command:
    full = f'@call {activate} pred-geomap & python {progr} "%~f0" %* & goto :eof'

    # save to file:
    with open("gui_main.bat", "w") as fout:
        fout.write(full)
except:
    print("Oops! Something went wrong.")
else: 
    print("Execution complete.")