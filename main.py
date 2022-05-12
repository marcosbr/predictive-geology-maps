"""Main executer script
"""
import argparse
import configparser
import os
from pathlib import Path

import time

from itertools import repeat
from collections.abc import Iterable 
import concurrent.futures

from predmap import PredMap

from osgeo import gdal
import numpy as np
from sklearn.metrics import classification_report

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

    return prediction

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
                     fnames_features, fname_target, fname_limit, dir_out[:-1],
                     target_field,
                     object_id,
                     discard_less_than, 
                     max_samples_per_class, 
                     use_coords,
                     use_cartesian_prod,
                     run_pca, 
                     pca_percent, 
                     rand_seed_num[:-1])
    
    # use a "regular call" to keep the object (multiprocessing does not share memory)
    pred = main(next(fnames_features), next(fname_target), next(fname_limit), 
                     dir_out[-1],
                     next(target_field),
                     next(object_id),
                     next(discard_less_than),
                     next(max_samples_per_class),
                     next(use_coords),
                     next(use_cartesian_prod),
                     next(run_pca),
                     next(pca_percent),
                     rand_seed_num[-1])
    
    # merge results:
    dir_out_root = os.path.dirname(dir_out[-1]) 
    y_pred_test = merge_results(dir_out_root).ravel()-1
    
    # compute metrics report for the averaged result
    # this uses the entire target raster
    msg = f'Metrics computed using the entire {next(fname_target)} rasterized.'
    ds = gdal.Open(os.path.join(dir_out[-1], f'{Path(next(fname_target)).resolve().stem}.tif'))
    y_test = ds.ReadAsArray().ravel()-1
    ds = None
    try:
        y_test = pred.le.inverse_transform(y_test)
        y_pred_test = pred.le.inverse_transform(y_pred_test)

        report = classification_report(y_test, y_pred_test)
        print(msg)
        print(report)

        with open(os.path.join(dir_out_root, 'classification_report.txt'), 
                    'w', encoding='utf-8') as fout:
            fout.write(msg)
            fout.write('\n')
            fout.write(report)

    except ValueError:
        print('Oops! Litologies were discarded during fit. Report does not contain original names.')
        print('Try adding more samples.')
   

def merge_results(dir_in):
    """Function to merge prediction results
    """
    # create one large list of all results (requires all data to fit in memory)
    res = []
    # get directories:
    experiments = next(os.walk(dir_in))[1]
    for experiment in experiments:
        experiment_prob = os.path.join(dir_in, experiment, 'class_probs.tif')
        experiment_class = os.path.join(dir_in, experiment, 'class.tif')
        if os.path.isfile(experiment_prob):
            ds = gdal.Open(experiment_prob)
            res.append(ds.ReadAsArray())
    stacked = np.stack(res) 
    probs_mean = np.mean(stacked, axis=0)

    # save a new raster with the result
    driver = gdal.GetDriverByName('GTiff')
    result = driver.CreateCopy(os.path.join(dir_in, 'class_probs_average.tif'), gdal.Open(experiment_prob))
    result.WriteArray(probs_mean)
    result = None

    # save class result:
    # add one to match band numbering (python starts from zero, bands start from 1)
    class_experiments = np.argmax(probs_mean, axis=0)+1
    driver = gdal.GetDriverByName('GTiff')
    result = driver.CreateCopy(os.path.join(dir_in, 'class_average.tif'), gdal.Open(experiment_class))
    result.WriteArray(class_experiments)
    result = None

    return class_experiments

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
