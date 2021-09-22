"""
Main class for the predictive geologicaal mapping
"""

from osgeo import gdal


class PredMap(gdal.Dataset):
    def __init__(self, fnames_features, fname_target, fname_limit):
        """[summary]

        Args:
            fnames_features (list): list of features filenames (rasters)
            fname_target (os.path): filename of the target (polygon vector layer)
            fname_limit (os.path): filename of the limiting boundary (polygon vector layer)
        """
        pass

    def rasterize(self):
        """Convert the target (vector) to raster
        """
        pass

    def resample(self):
        """Make sure all features have the same cell size
        """
        pass

    def clip(self):
        """Clip all features and target to the same limits
        """
        pass

    def set_raster_to_column(self):
        """Transform raster to numpy array to be used as feature
        """
        pass

    def set_target_to_column(self):
        """Transform raster to numpy array to be used as target
        """
        pass

    def fit(self):
        """Fit XGboost with grid search
        """
        pass

    def write_class_probs(self):
        """Write one multi-band raster containing all class probabilities
        """
        pass

    def write_class(self):
        """Write one single-band raster containing the class
        """
        pass

    def write_report(self):
        """Evaluate model and write the metrics
        (e.g., confusion matrix, classification report)
        """
        pass
