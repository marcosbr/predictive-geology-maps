"""A script to run them all

Following the new folder structure
"""

import concurrent.futures
import configparser
import os
import time
from pathlib import Path

base_config = 'config.ini'
location = 'Amapa'
results_dir = os.path.normpath(f'../data/{location}/resultados')

features_dir = os.path.normpath(f'../data/{location}/features_tif')
limits = os.path.normpath(f'../data/{location}/limites')
limit_tag = 'NA.22'

####################################################################
# read in the base config file:
config = configparser.ConfigParser()
config.read(base_config)


def process_region(folha):
    if limit_tag in folha and folha.endswith('.shp'):

        lim_name = os.path.join(limits, folha)
        folha = Path(folha).resolve().stem

        folha_dir_name = os.path.join(results_dir, folha)

        if not os.path.isdir(folha_dir_name):
            os.mkdir(folha_dir_name)

        print(folha)
        # change config.ini
        configfile = os.path.join(folha_dir_name, f'{folha}.ini')
        with open(configfile, 'w') as f:
            config['io']['fname_limit'] = os.path.abspath(lim_name)
            config['io']['dir_out'] = os.path.abspath(folha_dir_name)
            config.write(f)

        # execute
        command = f'python main.py -c={configfile}'
        os.system(command)


if __name__ == '__main__':
    t1 = time.perf_counter()

    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(process_region, os.listdir(limits))

    t2 = time.perf_counter()

    print(f'Finished in {(t2-t1)/60.:.2f} minutes')
