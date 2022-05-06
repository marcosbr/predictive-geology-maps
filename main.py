"""Main executer script
"""
import argparse
import configparser
import os
import time

from itertools import repeat
from collections.abc import Iterable 
import concurrent.futures

from predmap import PredMap


def main(fnames_features, fname_target, fname_limit, dir_out,
         target_field,
         object_id,
         discard_less_than, 
         max_samples_per_class, 
         use_coords,
         use_cartesian_prod,
         run_pca, 
         pca_percent=95.0, 
         rand_seed_num=0):
    """Main function
    """
    prediction = PredMap([os.path.normpath(fname) for fname in fnames_features],
                         os.path.normpath(fname_target),
                         os.path.normpath(fname_limit),
                         os.path.normpath(dir_out), 
                         target_field = target_field, 
                         object_id = object_id,
                         discard_less_than=int(discard_less_than),
                         max_samples_per_class=int(max_samples_per_class),
                         use_coords=use_coords, 
                         use_cartesian_prod=use_cartesian_prod,
                         run_pca=run_pca, 
                         pca_percent=float(pca_percent), 
                         rand_seed_num=int(rand_seed_num)
                         )

    prediction.fit()
    # the class probs can only be written if the model outputs
    # class probabilities
    prediction.write_class_probs()
    prediction.write_class('class.tif')
    prediction.write_class_vector()

def make_iterables(**kwargs):
    """Function that checks all entries in the dictionary and returns
    an iterable version of the value when the entry is not iterable.
    Example:
    ```
    print(arg1=2,arg2=[1,1,1], arg3='foo')
    >>> {'arg1': repeat(2), 'arg2': [1, 1, 1], 'arg3': repeat('foo')}
    ```
    """
    for key, val in kwargs.items():
        if not isinstance(val, Iterable) or type(val)==str:
            kwargs[key] = repeat(val)
    return kwargs

def multiple_realizations(fnames_features, fname_target, fname_limit, dir_out,
         target_field,
         object_id,
         discard_less_than, 
         max_samples_per_class, 
         use_coords,
         use_cartesian_prod,
         run_pca, 
         pca_percent, 
         rand_seed_num):
    """
    Threaded call to main using the same arguments. 
    Arguments are expected to be iterables.
    """     
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(main, 
                     fnames_features, fname_target, fname_limit, dir_out,
                     target_field,
                     object_id,
                     discard_less_than, 
                     max_samples_per_class, 
                     use_coords,
                     use_cartesian_prod,
                     run_pca, 
                     pca_percent, 
                     rand_seed_num
        )

if __name__ == '__main__':

    start_time = time.perf_counter()
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config_file', default='config.ini')

    args = parser.parse_args()

    if os.path.isfile(args.config_file):
        config = configparser.ConfigParser()
        config.read(args.config_file)

        main(config['io']['fnames_features'].split('\n'),
             config['io']['fname_target'],
             config['io']['fname_limit'],
             config['io']['dir_out'], 
             config['options']['target_field'], 
             config['options']['object_id'], 
             config['options']['discard_less_than'], 
             config['options']['max_samples_per_class'],
             config['options']['use_coords'],
             config['options']['use_cartesian_prod'],
             config['options']['run_pca'],
             config['options']['pca_percent'],
             config['advanced']['rand_seed_num'])
            

    else:
        print("Please provide a valid configuration file")

    end_time = time.perf_counter()

    print(f'\nExecution time: {(end_time-start_time)/60:.2f} minutes')
