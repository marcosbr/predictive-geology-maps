"""
Main class for the predictive geological mapping
"""
import os
from pathlib import Path

import numpy as np
from osgeo import gdal, osr
from tqdm import tqdm


class PredMap():
    """Main class
    """

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

        # these will be assembled by the class
        self.X = None
        self.y = None
        self.y_pred = None
        self.target_raster = None

        # check if the output directory exists
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
        # assumes cells are square (cell size x == cell size y)
        self.lower_res_idx = np.argmax(
            [raster.GetGeoTransform()[1] for raster in self.feature_rasters])

        # keep the low resolution raster easily accessible:
        self.lowres = self.feature_rasters[self.lower_res_idx]

        # setup the class projection:
        self.proj = osr.SpatialReference()
        self.proj.ImportFromWkt(self.lowres.GetProjectionRef())

        # rasterize target (also clips according to fname_limit):
        self.rasterize()

        # resample everyone (also clips according to fname_limit):
        self.resample()

        # update the class projection:
        self.proj = osr.SpatialReference()
        self.proj.ImportFromWkt(self.target_raster.GetProjectionRef())

        # set up self.X and self.y
        self.set_rasters_to_column()

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

        # close to write the raster
        rasterized = None

        # close the target shape
        self.target = None

        # clip raster:
        self.clip(target_raster_fname, temp_raster_fname)

        # we want the target raster to be acessible:
        self.target_raster = gdal.Open(target_raster_fname)

        # delete temporary raster:
        os.remove(temp_raster_fname)

    def resample(self):
        """Make sure all features have the same cell size
        """

        temp_raster_fname = os.path.join(self.dir_out, 'temp.tif')

        for fname, (idx, raster) in tqdm(zip(self.fnames_features,
                                             enumerate(self.feature_rasters)),
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

            # close to write the raster
            dest = None

            # clip rasters:
            self.clip(feature_resampled_fname, temp_raster_fname)

            # read back in the new resampled and clipped raster
            self.feature_rasters[idx] = gdal.Open(feature_resampled_fname)

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

    def set_rasters_to_column(self):
        """Transform raster to numpy array to be used for training and eval
        """
        feats = []
        for raster in self.feature_rasters:
            for idx in range(raster.RasterCount):
                # Read the raster band as separate variable
                band = raster.GetRasterBand(idx+1)
                # get numpy vals
                band_np = band.ReadAsArray()
                # mark NaNs
                band_np[band.GetMaskBand().ReadAsArray() == 0] = np.nan

                feats.append(np.reshape(band_np, (-1, )))

        self.X = np.array(feats).T

        # set up target array
        # Read the raster band as separate variable
        band = self.target_raster.GetRasterBand(1)
        # get numpy vals
        band_np = band.ReadAsArray()
        # assure band_np is float and not int to be able to asign np.nan
        # should we do this or define an "int nan"?
        band_np = band_np.astype(float)
        # mark NaNs
        band_np[band.GetMaskBand().ReadAsArray() == 0] = np.nan

        self.y = np.reshape(band_np, (-1, 1))

    def fit(self):
        """Fit XGboost with grid search
        """
        # TODO: for now, just create an output
        self.y_pred = np.random.randn(self.y.shape[0],
                                      len(np.unique(self.y)))

    def write_class_probs(self):
        """Write one multi-band raster containing all class probabilities
        """
        temp_raster_fname = os.path.join(self.dir_out, 'class_probs.tif')
        # create an in-memory raster
        drv_tiff = gdal.GetDriverByName('GTiff')
        dest = drv_tiff.Create(temp_raster_fname,
                               self.target_raster.RasterXSize,
                               self.target_raster.RasterYSize,
                               self.y_pred.shape[1],
                               gdal.GDT_Float32)

        dest.SetGeoTransform(self.target_raster.GetGeoTransform())
        dest.SetProjection(self.proj.ExportToWkt())

        for idx in range(dest.RasterCount):
            band = dest.GetRasterBand(idx+1)
            out = self.y_pred[:, idx]
            out = np.reshape(out, (self.target_raster.RasterXSize,
                                   self.target_raster.RasterYSize))
            band.WriteArray(out)

        # close to write the raster
        dest = None

    def write_class(self):
        """Write one single-band raster containing the class
        """
        temp_raster_fname = os.path.join(self.dir_out, 'class.tif')
        # create an in-memory raster
        drv_tiff = gdal.GetDriverByName('GTiff')
        dest = drv_tiff.Create(temp_raster_fname,
                               self.target_raster.RasterXSize,
                               self.target_raster.RasterYSize,
                               1,
                               gdal.GDT_Float32)

        dest.SetGeoTransform(self.target_raster.GetGeoTransform())
        dest.SetProjection(self.proj.ExportToWkt())

        band = dest.GetRasterBand(1)
        out = np.argmax(self.y_pred, axis=1)
        out = np.reshape(out, (self.target_raster.RasterXSize,
                               self.target_raster.RasterYSize))
        band.WriteArray(out)

        # close to write the raster
        dest = None

    def write_report(self):
        """Evaluate model and write the metrics
        (e.g., confusion matrix, classification report)
        """
        pass
