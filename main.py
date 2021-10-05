"""Main executer script
"""
import argparse
import configparser
import os
import time

from predmap import PredMap


def main(fnames_features, fname_target, fname_lab_conv, fname_limit, dir_out):
    """Main function
    """
    print(fnames_features)
    print(fname_target)
    print(fname_limit)
    print(dir_out)

    prediction = PredMap([os.path.normpath(fname) for fname in fnames_features],
                         os.path.normpath(fname_target),
                         os.path.normpath(fname_lab_conv),
                         os.path.normpath(fname_limit),
                         os.path.normpath(dir_out))

    prediction.fit()
    # the class probs can only be written if the model outputs
    # class probabilities
    prediction.write_class_probs()
    prediction.write_class()


if __name__ == '__main__':

    start_time = time.perf_counter()
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config_file', default='config.ini')

    parser.add_argument('-fin_feat', '--fnames_features', default=None)
    parser.add_argument('-fin_targ', '--fname_target', default=None)
    parser.add_argument('-fin_lim', '--fname_limit', default=None)

    parser.add_argument('-d_out', '--dir_out', default=None)

    args = parser.parse_args()

    if os.path.isfile(args.config_file):
        config = configparser.ConfigParser()
        config.read(args.config_file)

        main(config['io']['fnames_features'].split('\n'),
             config['io']['fname_target'],
             config['io']['fname_lab_conv'],
             config['io']['fname_limit'],
             config['io']['dir_out'])

    else:
        main(args.fnames_features,
             args.fname_target,
             args.fname_limit,
             args.dir_out)

    end_time = time.perf_counter()

    print(f'\nExecution time: {(end_time-start_time)/60:.2f} minutes')
