"""
Main class for the predictive geological mapping
"""
import os
from pathlib import Path

import numpy as np
from osgeo import gdal, osr
from tqdm import tqdm


class PredMap():
    def __init__(self,
                 fnames_features,
                 fname_target,
                 fname_limit,
                 dir_out):
        """[summary]

        Args:
            fnames_features (list): list of features filenames (rasters)
            fname_target (os.path - file): filename of the target (polygon vector layer)
            fname_limit (os.path - file): filename of the limiting boundary (polygon vector layer)
            dir_out (os.path - directory): directory where the output files will be saved
        """
        self.fnames_features = fnames_features
        self.fname_target = fname_target
        self.fname_limit = fname_limit
        self.dir_out = dir_out

        # these will be assembled
        self.X = None
        self.y = None
        self.target_raster = None

        if not os.path.isdir(dir_out):
            os.mkdir(dir_out)

        # read all input rasters:
        self.feature_rasters = [gdal.Open(fname) for fname in fnames_features]

        # read target shape
        self.target = gdal.OpenEx(fname_target)

        # are all elements in the same projection?
        look_at = [raster.GetProjectionRef()
                   for raster in self.feature_rasters]
        look_at.append(self.target.GetProjectionRef())

        if len(set(look_at)) != 1:
            print("Warning-Input files do not have the same Projection")
            [print(f'{it}\n') for it in look_at]

        # get the worst resolution raster
        # assumes cells are square
        self.lower_res_idx = np.argmax(
            [raster.GetGeoTransform()[1] for raster in self.feature_rasters])

        # keep that raster easily accessible:
        self.lowres = self.feature_rasters[self.lower_res_idx]

        # setup the projection:
        self.proj = osr.SpatialReference()
        self.proj.ImportFromWkt(self.lowres.GetProjectionRef())

        # rasterize target:
        self.rasterize()

        # resample everyone:
        self.resample()

    def rasterize(self):
        """Convert the target (vector) to raster
        """
        # get the layer shape
        lyr = self.target.GetLayer()

        temp_raster_fname = os.path.join(self.dir_out, 'temp.tif')
        target_raster_fname = os.path.join(self.dir_out,
                                           f'{Path(self.fname_target).resolve().stem}.tif')

        # setup a new raster
        drv_tiff = gdal.GetDriverByName("GTiff")

        rasterized = drv_tiff.Create(temp_raster_fname,
                                     self.lowres.RasterXSize, self.lowres.RasterYSize,
                                     1,
                                     gdal.GDT_Int16)
        rasterized.SetGeoTransform(self.lowres.GetGeoTransform())
        rasterized.SetProjection(self.proj.ExportToWkt())

        # set the "No Data Value"
        nanval = -9999
        rasterized_band = rasterized.GetRasterBand(1)
        rasterized_band.Fill(nanval)
        rasterized.GetRasterBand(1).SetNoDataValue(nanval)

        # rasterize the shape
        # needs numeric attribute!
        # TODO: the attribute should be an option variable, not hardcoded!
        gdal.RasterizeLayer(rasterized, [1], lyr,
                            options=["ALL_TOUCHED=TRUE",
                                     "ATTRIBUTE=OBJECTID"])

        rasterized.GetRasterBand(1).SetNoDataValue(nanval)
        # close to write the raster
        rasterized = None

        # close the target shape
        self.target = None

        # clip rasters:
        self.clip(target_raster_fname, temp_raster_fname)

        # we want the target raster:
        self.target_raster = gdal.Open(target_raster_fname)

        # delete temporary raster:
        os.remove(temp_raster_fname)

    def resample(self):
        """Make sure all features have the same cell size
        """

        temp_raster_fname = os.path.join(self.dir_out, 'temp.tif')

        for fname, raster in tqdm(zip(self.fnames_features, self.feature_rasters),
                                  desc='Resampling Rasters'):

            feature_resampled_fname = os.path.join(self.dir_out,
                                                   f'{Path(fname).resolve().stem}.tif')

            # create an in-memory raster
            drv_tiff = gdal.GetDriverByName('GTiff')
            dest = drv_tiff.Create(temp_raster_fname,
                                   self.lowres.RasterXSize, self.lowres.RasterYSize,
                                   raster.RasterCount,
                                   gdal.GDT_Float32)
            dest.SetGeoTransform(self.lowres.GetGeoTransform())
            dest.SetProjection(self.proj.ExportToWkt())

            gdal.ReprojectImage(raster, dest,
                                raster.GetProjectionRef(), self.proj.ExportToWkt(),
                                gdal.GRA_Bilinear)
            dest = None

            # clip rasters:
            self.clip(feature_resampled_fname, temp_raster_fname)

        # delete temporary raster:
        os.remove(temp_raster_fname)

    def clip(self, fname_out, fname_in):
        """Clip all features and target to the same limits

        Args:
            fname_out (os.path): input file name
            fname_in (os.path): output file name
        """
        gdal.Warp(fname_out, fname_in,
                  cutlineDSName=self.fname_limit,
                  cropToCutline=True)

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
