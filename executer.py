"""A script to run them all
"""

import concurrent.futures
import configparser
import os
import time

base_config = 'config.ini'
results_dir = os.path.normpath('data/results')

features_dir = os.path.normpath('data/features_tif')
lito_1_250k_dir = os.path.normpath('data/Litologia_Rio Maria_1_250.000')
lito_1_100k_dir = os.path.normpath(
    'data/Arquivos_Geologia_1_100_000_Rio_Maria')
# read in the base config file:
config = configparser.ConfigParser()
config.read(base_config)


def process_region(folha):
    # for folha in os.listdir(lito_1_100k_dir):
    print(folha)
    for folha_in in os.listdir(os.path.join(lito_1_100k_dir, folha)):
        print(folha_in)
        plan = os.path.join(lito_1_100k_dir, folha,
                            folha_in, 'Planimetria')
        lim_name = [lim for lim in os.listdir(
            plan) if "Limite_da_Folha_A" in lim]
        lim_name = [lim for lim in os.listdir(
            plan) if lim.endswith('.shp')]
        lim_name = os.path.join(plan, lim_name[0])

        # change config.ini
        with open(f'{folha}.ini', 'w') as configfile:
            config['io']['fname_limit'] = os.path.abspath(lim_name)
            config['io']['dir_out'] = os.path.abspath(
                os.path.join(results_dir, folha))
            config.write(configfile)

        # execute
        command = f'python main.py -c={folha}.ini'
        os.system(command)


if __name__ == '__main__':
    t1 = time.perf_counter()

    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(process_region, os.listdir(lito_1_100k_dir))

    t2 = time.perf_counter()

    print(f'Finished in {(t2-t1)/60.:2} minutes')
