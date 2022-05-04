"""Main executer script
"""
import argparse
import configparser
import os
import time

from predmap import PredMap


def main(fnames_features, fname_target, fname_limit, dir_out,
         target_field,
         object_id,
         discard_less_than, 
         max_samples_per_class, 
         use_coords,
         use_crosses,
         run_pca, 
         pca_percent=95):
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
                         use_crosses=use_crosses,
                         run_pca=run_pca, 
                         pca_percent=float(pca_percent))

    prediction.fit()
    # the class probs can only be written if the model outputs
    # class probabilities
    prediction.write_class_probs()
    prediction.write_class('class.tif')
    prediction.write_class_vector()


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
             config['options']['use_crosses'],
             config['options']['run_pca'],
             config['options']['pca_percent'])
            

    else:
        print("Please provide a valid configuration file")

    end_time = time.perf_counter()

    print(f'\nExecution time: {(end_time-start_time)/60:.2f} minutes')
