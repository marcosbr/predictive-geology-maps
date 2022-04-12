"""
Main class for the predictive geological mapping
"""

import sys
import itertools
import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE

from osgeo import gdal, ogr, osr
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from tqdm import tqdm
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

class PredMap():
    """Main class
    """

    def __init__(self,
                 fnames_features,
                 fname_target,
                 fname_limit,
                 dir_out, 
                 target_field,
                 object_id,
                 discard_less_than, 
                 max_samples_per_class,
                 use_coords,
                 run_pca, 
                 pca_percent=95.0):
        """[summary]

        Args:
            fnames_features (list): list of features filenames (rasters)
            fname_target (os.path - file): filename of the target (polygon vector layer)
            fname_limit (os.path - file): filename of the limiting boundary (polygon vector layer)
            target_field (string): field name of the unique identifier (fiducial)
            object_id (string): field name of the target attribute that will be predicted
            dir_out (os.path - directory): directory where the output files will be saved
            discard_less_than (integer): discard categories with fewer than this number of samples
            max_samples_per_class (integer): maximum number of samples per class to keep (random resample)
            use_coords (boolean): set to True to use coordinates as predictors (features)
            run_pca (boolean): set to True to use PCA to reduce dimensionality of multi-band rasters
            pca_percent (float): percentage of the variance to keep when pca is selected
        """
        self.fnames_features = fnames_features
        self.fname_target = fname_target
        self.fname_limit = fname_limit
        self.dir_out = dir_out
        self.discard_less_than = discard_less_than
        self.max_samples_per_class = max_samples_per_class

        # integer value to be used as nan
        self.nanval = -9999

        # target attribute:
        self.target_attribute = target_field
        # integer identifier
        self.object_id = object_id

        # these will be assembled by the class
        self.X = None
        self.y = None
        self.y_pred = None
        self.target_raster = None
        self.le_df = None
        self.lab_to_int = None
        self.int_to_lab = None
        self.target_raster_fname = None
        self.fname_lab_conv = None
        self.list_of_features = []
        self.dataframe = None
        self.run_pca = run_pca
        self.pca_percent = pca_percent
        self.use_coords = use_coords
        self.list2pca = [] # list of names of multi-band rasters
        self.nan_mask = None
        self.le = LabelEncoder()


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

        # create the dictionary:
        self.create_unique_litos()

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
        # create a dictionary mapping OBJECTID to target
        obj_sigla_dict = {}

        for feature in lyr:
            obj_sigla_dict[feature.GetField(
                self.object_id)] = feature.GetField(self.target_attribute)

        # set up raster names
        temp_raster_fname = os.path.join(self.dir_out, 'temp.tif')
        target_raster_fname = os.path.join(self.dir_out,
                                           f'{Path(self.fname_target).resolve().stem}.tif')
        self.target_raster_fname = target_raster_fname

        # setup a new raster
        drv_tiff = gdal.GetDriverByName("GTiff")

        rasterized = drv_tiff.Create(temp_raster_fname,
                                     self.lowres.RasterXSize, self.lowres.RasterYSize,
                                     1,
                                     gdal.GDT_Int16)
        rasterized.SetGeoTransform(self.lowres.GetGeoTransform())
        rasterized.SetProjection(self.proj.ExportToWkt())

        # set the "No Data Value"
        rasterized_band = rasterized.GetRasterBand(1)
        rasterized_band.Fill(self.nanval)
        rasterized.GetRasterBand(1).SetNoDataValue(self.nanval)

        # rasterize the shape
        # needs numeric attribute!
        gdal.RasterizeLayer(rasterized, [1], lyr,
                            options=["ALL_TOUCHED=TRUE",
                                     f"ATTRIBUTE={self.object_id}"])

        # close to write the raster
        rasterized = None

        # close the target shape
        self.target = None

        # clip raster:
        self.clip(target_raster_fname, temp_raster_fname)

        # reopen raster to replace OBJECTID by lithology:
        self.target_raster = gdal.Open(target_raster_fname, 1)
        band = self.target_raster.GetRasterBand(1)
        band_np = band.ReadAsArray()

        # replace OBJECTID by sigla
        out = np.empty(band_np.shape, dtype='U25')
        for key, val in obj_sigla_dict.items():
            idx = band_np == key
            out[idx] = val

        # label encoding for the project:
        self.le_df = pd.read_csv(self.fname_lab_conv)
        self.lab_to_int = dict(zip(self.le_df[self.target_attribute],
                                   self.le_df['VALUE']))
        self.int_to_lab = dict(zip(self.le_df['VALUE'],
                                   self.le_df[self.target_attribute]))

        out = pd.Series(out.ravel()).map(
            self.lab_to_int).to_numpy().reshape(band_np.shape)
        out[band.GetMaskBand().ReadAsArray() == 0] = self.nanval
        band.WriteArray(out)

        # write array
        self.target_raster = None

        # we want the target raster to be acessible:
        self.target_raster = gdal.Open(target_raster_fname)

        # delete temporary raster:
        os.remove(temp_raster_fname)

    def resample(self):
        """Make sure all rasters have the same cell size
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

            # set the "No Data Value"
            band = dest.GetRasterBand(1)
            band.Fill(self.nanval)
            dest.GetRasterBand(1).SetNoDataValue(self.nanval)

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
            fname_out (os.path): input file name (raster)
            fname_in (os.path): output file name (raster)
        """
        gdal.Warp(fname_out, fname_in,
                  cutlineDSName=self.fname_limit,
                  cropToCutline=True)

    def set_rasters_to_column(self):
        """
            Transform raster to numpy array to be used for training and eval
        """

        feats = []
        raster = self.feature_rasters[0]
        self.dataframe = pd.DataFrame.from_records(itertools.product(range(raster.RasterYSize),
                                                            range(raster.RasterXSize)), columns=['Row', 'Column'])

        for raster, raster_name in zip(self.feature_rasters, self.fnames_features):
            for idx in range(raster.RasterCount):
                # Read the raster band as separate variable
                band = raster.GetRasterBand(idx+1)
                # get numpy vals
                band_np = band.ReadAsArray()
                band_np = band_np.astype(float)
                feats.append(np.reshape(band_np, (-1, )))

                # Separate rasters with single and multilayer
                if raster.RasterCount == 1:
                    colname = Path(raster_name).resolve().stem
                    self.dataframe[colname] = np.reshape(band_np, (-1, 1))
                elif raster.RasterCount > 1:
                    rst = np.nan_to_num(band_np, nan=self.nanval)
                    colname = Path(raster_name).resolve().stem + '..band' + str(idx)
                    self.dataframe[colname] = np.reshape(rst, (-1, 1))

                self.list_of_features.append(colname)

            if raster.RasterCount > 1:
                # Put the prefix name of column to all multiraster to run PCA
                self.list2pca.append(Path(raster_name).resolve().stem)

        self.X = np.array(feats).T

        # check for null values in the feature columns:
        self.nan_mask = self.dataframe.isin([self.nanval]).any(axis=1)

        # set up target array
        # Read the raster band as separate variable
        band_np = self.target_raster.ReadAsArray()
        self.y = np.reshape(band_np, (-1, 1))

        self.dataframe['TARGET'] = self.y
        # we can keep the original names in the dataframe:
        self.dataframe['TARGET'] = self.dataframe['TARGET'].map(self.int_to_lab)
        # make sure to use the program's "null value"
        self.dataframe['TARGET'] = self.dataframe['TARGET'].fillna(self.nanval)

        if not self.use_coords:
            self.dataframe = self.dataframe.drop(['Row', 'Column'], axis=1)
        if self.run_pca:
            self.dim_reduct()
    
    def dim_reduct(self):
        """
        User might want to use dimensionality reduction. 
        As this is an unsupervised technique, we can fit_transform
        the entire raster - as we would if we pre-processed the
        raster using a GIS software (using all pixels). This means 
        the data is pre-processed before data split (train, val, test).
        This should be ok in this case as all input rasters will have
        pixel values everywhere. 
        """

        # sklearn's PCA center's the data, but does not scale it. 
        # so we use a standard scaler before PCA:
        std_scaler = StandardScaler()
        pca = PCA()

        # loop through multi-band rasters:
        for mband in self.list2pca:
            mask = self.dataframe.columns.str.startswith(f'{mband}..band')
            # select marked columns:
            X = self.dataframe.loc[:, mask].to_numpy()
            # zero center and scale it:
            X = std_scaler.fit_transform(X)
            # project:
            pcs = pca.fit_transform(X)
            # find enough pcs to keep the desired explained variance:
            var_exp = np.cumsum(pca.explained_variance_ratio_)
            keep = np.searchsorted(var_exp, self.pca_percent/100.0)+1
            print(f'Program will keep the first {keep} PCs of {mband}')
            # replace the projected onto the dataframe:
            # first, drop the bands
            self.dataframe = self.dataframe.loc[:, ~mask]
            # add the new projected column
            pc_df = pd.DataFrame(pcs[:, :keep])
            # pcs name should start from 1, not from zero:
            pc_df.columns += 1
            # and we want to know where they come from:
            pc_df = pc_df.add_prefix(f'{mband}-PC_')
            # finally, merge it with the dataframe:
            self.dataframe = pd.concat([self.dataframe, pc_df], axis=1)

    def get_single_raster_features(self):
        idxs_multiband = []
        for prefix in list(set(self.list2pca)):
            idxs_multiband.append(self.get_columns2pca(prefix))
        idxs_multiband = [item for sublist in idxs_multiband for item in sublist]

        idxs_singleband = []
        for i in np.arange(len(self.list_of_features)):
            if i not in idxs_multiband:
                idxs_singleband.append(i)
        return idxs_singleband

    def fit(self):
        """
        Fit XGboost with Randomized search
        """

        df_original = self.dataframe.copy()
        df = self.dataframe.copy()

        # drop all nan vals:
        nan_mask = df.isin([self.nanval]).any(axis=1)

        print(f'Before dropping nan values: {df.shape}')
        df = df[~nan_mask]
        print(f'After dropping nan values: {df.shape}')
        self.nan_mask = nan_mask

        lito_count = pd.DataFrame(df.TARGET.value_counts())
        lito_count.rename({'TARGET':'Count'}, axis=1, inplace=True)
        lito_count['Discard'] = lito_count['Count'] < self.discard_less_than

        df = df[~df.TARGET.isin(lito_count.loc[lito_count['Discard']==True].index.to_list())]

        print('Discarded targets')
        print(lito_count.to_string())

        # fit the label encoder 
        self.le.fit(df.TARGET.ravel())
        
        #########################################
        # training data selection
        #########################################
        # create individual number of samples per class
        samples_per_group_dict = {}
        lito_count = df.TARGET.value_counts()
        for index, value in lito_count.items():
            samples_per_group_dict[index] = np.min([self.max_samples_per_class, value])

        # select the number of samples per class 
        list_of_sampled_groups = []
        for name, group in df.groupby('TARGET'):    
            n_rows_to_sample = samples_per_group_dict[name]
            sampled_group = group.sample(n_rows_to_sample)
            list_of_sampled_groups.append(sampled_group)

        # get the undersampled dataset
        df_under = pd.concat(list_of_sampled_groups)

        # the remaining data serves as test set:
        df_test = df.drop(df_under.index, axis=0)
        if len(df_test) == 0:
            # if the number of samples is small (when using stations), 
            # it is possible that the test dataframe is empty
            # so we use the whole dataset again
            df_test = df.copy()
            # save a message to be printed 
            msg = 'Small dataset - test data is the target dataset after discarding' \
                + f'classes with fewer than {self.discard_less_than} samples.'
        else:
            msg = f'Test data with {len(df_test)} samples was randomly selected before training'

        # delete temporary list to release memory
        del list_of_sampled_groups

        # print some messages:
        print(f'Train dataframe shape: {df_under.shape}')
        print(f'Test dataframe shape: {df_test.shape}')

        #########################################
        # pre-processing
        #########################################
        # Unlike the dimensionality reduction step, here we need to
        # make sure the models are fit only on the training data. 
        # The trained models will then be used to transform the
        # test data (and the full dataframe). 

        # --------------------define models-------------------------------:
        std_scaler = StandardScaler()
        oversamp = SMOTE(random_state=0)
        clf = XGBClassifier(eval_metric='mlogloss', random_state=42)

        xgb_param = [{'booster': ['gbtree', 'gblinear', 'dart'],
                      'learning_rate': [0.1, 0.2, 0.3, 0.5, 0.6, 0.8, 0,95],
                      'max_depth': [2, 3, 6, 10, 20],
                      'min_child_weight': [0, 1, 5, 10, 50],
                      'max_delta_step': [0, 1, 5, 10, 50],
                      'subsample': [0, 0.5, 1],
                      'sampling_method': ['uniform', 'gradient_based']}]
        
        clf_search = RandomizedSearchCV(clf, 
                                        xgb_param, 
                                        random_state=0, 
                                        verbose=3, 
                                        n_iter=10)

        # ----------------continue with data prep-------------------------:
        X_train = df_under.drop(['TARGET'], axis=1).to_numpy()
        y_train = df_under.loc[:, 'TARGET'].to_numpy()
        y_train = self.le.transform(y_train.ravel())

        # scale data
        std_scaler.fit(X_train)
        X_train = std_scaler.transform(X_train)

        # oversample the training set:
        X_train, y_train = oversamp.fit_resample(X_train, y_train)

        #########################################
        # hyperparameter tuning and fit
        #########################################
        search = clf_search.fit(X_train, y_train)
        print(f'Mean cross-validated score of the best_estimator: {search.best_score_:.2f}')

        #########################################
        # using the trained model
        #########################################

        # ----------------test set-------------------------:
        # use the trained model on the hold-out test set:
        X_test = df_test.drop(['TARGET'], axis=1).to_numpy()
        y_test = df_test.loc[:, 'TARGET'].to_numpy()

        # scale data
        std_scaler.fit(X_test)
        X_test = std_scaler.transform(X_test)

        # predict
        y_pred_test = search.predict_proba(X_test)

        # save performance information:
        y_pred_test = self.le.inverse_transform(np.argmax(y_pred_test, axis=1))

        report = classification_report(y_test, y_pred_test)
        print(msg)
        print(report)

        with open(os.path.join(self.dir_out, 'classification_report.txt'), 
                   'w', encoding='utf-8') as fout:
            fout.write(msg)
            fout.write('\n')
            fout.write(report)
        
        # write the encoder:
        le_out_df = pd.DataFrame(self.le.classes_, columns=[self.target_attribute])
        le_out_df['Band'] = self.le.transform(le_out_df[self.target_attribute])+1
        le_out_df.to_csv(os.path.join(self.dir_out, f'{self.target_attribute}-to-band.csv'), 
                         index=False)

        # ----------------full data-------------------------:
        # use the trained model on the full data:
        df_original = df_original.fillna(0)
        
        X = df_original.drop(['TARGET'], axis=1).to_numpy()
        X = std_scaler.transform(X)

        self.y_pred = search.predict_proba(X)

    def write_class_probs(self):
        """
        Write one multi-band raster containing all class probabilities
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
            out = self.y_pred[:, idx].astype(np.float32)
            out = np.reshape(out, (self.target_raster.RasterYSize,
                                   self.target_raster.RasterXSize))

            band.WriteArray(out)

            # set the "No Data Value"
            band.SetNoDataValue(self.nanval)

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
                               gdal.GDT_Int16)

        dest.SetGeoTransform(self.target_raster.GetGeoTransform())
        dest.SetProjection(self.proj.ExportToWkt())

        band = dest.GetRasterBand(1)
        # add one to match band numbering (python starts from zero, bands start from 1)
        out = np.argmax(self.y_pred, axis=1)+1
        # reshape to raster dimensions
        out = np.reshape(out, (self.target_raster.RasterYSize,
                               self.target_raster.RasterXSize))

        band.WriteArray(out)

        # set the "No Data Value"
        band.SetNoDataValue(self.nanval)

        # close to write the raster
        dest = None

        self.geological_color()

    def write_class_vector(self):
        """Vectorize the resulting class raster
        """
        # check to see if class.tif was written:
        ds_pred = gdal.Open(os.path.join(self.dir_out, 'class.tif'))
        if ds_pred is None:
            print("Unable to open predicted raster.")
            return
        
        band = ds_pred.GetRasterBand(1)
        dst_layername = "Prediction"
        
        # set up the shapefile driver
        drv = ogr.GetDriverByName("ESRI Shapefile")
        # create the data source
        dst_ds = drv.CreateDataSource(os.path.join(self.dir_out, "class.shp"))
        # create the layer
        dst_layer = dst_ds.CreateLayer(dst_layername, srs= self.proj)
        
        pred_field = ogr.FieldDefn('Prediction', ogr.OFTInteger)
        dst_layer.CreateField(pred_field)
        gdal.Polygonize(band, None, dst_layer, 0, [], callback=None )
    
        # convert integer to lythology
        pred_field = ogr.FieldDefn(self.target_attribute, ogr.OFTString)
        dst_layer.CreateField(pred_field)

        for feat_idx in range(dst_layer.GetFeatureCount()):
            feat = dst_layer.GetFeature(feat_idx)
            if  feat.GetField("Prediction") == self.nanval:
                continue
            feat.SetField(self.target_attribute, 
                        # we have to -1 to compensate band to python (band starts at 1, python at 0)
                        self.le.inverse_transform([feat.GetField("Prediction")-1]).item())
            dst_layer.SetFeature(feat)

    def geological_color(self):
        '''
         Function to write a file in QGIS format with 'geological' colors
         with the lithology unique symbology

         unique_litos: csv file with unique lithology
         class_value: csv file with the classification results for each region
        '''

        def scale(PP, aux, litos):
            '''aux: length of litos'''
            colors = {}
            step1 = np.round(np.linspace(
                PP['min'][0], PP['max'][0], aux) * 255).astype(int)
            step2 = np.round(np.linspace(
                PP['min'][1], PP['max'][1], aux) * 255).astype(int)
            step3 = np.round(np.linspace(
                PP['min'][2], PP['max'][2], aux) * 255).astype(int)
            colors['r'] = step1
            colors['g'] = step2
            colors['b'] = step3
            colors['a'] = 255 * np.ones(len(step1)).astype(int)
            colors[self.target_attribute] = litos
            df = pd.DataFrame.from_dict(colors)
            return df

        def _returnUpper(text):
            output_string = ""
            for character in text:
                if character.isupper():
                    output_string += character
            return output_string

        def count(litos, gtime):
            ''' Count number of litologies in each geological time, returning the number and a
                list with the correspond litologies
            '''
            aux = 0
            litologias = []
            for t in litos:
                if gtime == _returnUpper(t):
                    aux += 1
                    litologias.append(t)
            return aux, litologias

        table_color = {'A': {'min': np.array([235, 169, 184]) / 255, 'max': np.array([226, 118, 158]) / 255},
                       'PP': {'min': np.array([244, 173, 201]) / 255, 'max': np.array((200, 36, 93)) / 255},
                       'MP': {'min': np.array([246, 200, 167]) / 255, 'max': np.array([190, 90, 35]) / 255},
                       'NP': {'min': np.array([250, 206, 128]) / 255, 'max': np.array([244, 184, 107]) / 255},
                       'NQ': {'min': np.array([255, 255, 0]) / 255, 'max': np.array([255, 241, 114]) / 255},
                       'N': {'min': np.array([250, 206, 128])/255, 'max': np.array([255, 241, 114])/255},
                       'E': {'min': np.array([226, 182, 119])/255, 'max': np.array([234, 177, 95])/255},
                       'Q': {'min': np.array([255, 255, 0]) / 255, 'max': np.array([251, 227, 220]) / 255},
                       '': {'min': np.array([255, 255, 0]) / 255, 'max': np.array([251, 227, 220]) / 255}}

        file = os.path.join(self.dir_out, 'class.tif')
        litos2 = pd.read_csv(os.path.join(self.dir_out, f'{self.target_attribute}-to-band.csv'))

        a = 0
        ds = gdal.Open(file)
        band = ds.GetRasterBand(1)
        array = np.array(band.ReadAsArray())
        values = np.unique(array)
        ds = None

        litos = litos2[self.target_attribute]
        ids = list(values)

        for v in values:
            if v == self.nanval:
                ids.remove(self.nanval)
                continue
            if v == -32768:
                ids.remove(-32768)
                continue

            a += 1

        ldf = []

        for t in ['Q', 'NP', 'NQ', 'N', 'MP',  'E', 'PP', 'A']:
            try:
                aux, sl_litos = count(litos, t)
                ldf.append(scale(table_color[t], aux, sl_litos))
            except:
                continue

        df = pd.concat(ldf)
        try:
            df['ID'] = ids
        except:
            print('Warning!!! Geological units not yet mapped to create geological color!\n'
                  'Please contact the developers.')
            return 
        outfile = os.path.join(self.dir_out, 'color.csv')
        df = df.reindex(columns=['ID', 'r', 'g', 'b', 'a', self.target_attribute])
        df.to_csv(outfile, index=False, header=False)

        # write .clr file 
        outfile = os.path.join(self.dir_out, 'color.clr')
        df = df.reindex(columns=['ID', 'r', 'g', 'b'])
        df.to_csv(outfile, index=False, header=False, sep=' ')

    def create_unique_litos(self):
        '''
            Create unique labels of geology
            write self.target_attribute.csv
        '''

        ds = ogr.Open(self.fname_target, 0)
        if ds is None:
            sys.exit('Could not open {0}.'.format(self.fname_target))
        lyr = ds.GetLayer(0)
        litos = []
        for feat in lyr:
            pt = feat.geometry()
            name = feat.GetField(self.target_attribute)
            if name not in litos:
                litos.append(name)
        ds = None
        litos.sort()

        ids = list(np.arange(len(litos)) + 1)
        temp = {self.target_attribute: litos,
                'VALUE': ids}

        fname_lab_conv = os.path.join(self.dir_out, f'{self.target_attribute}.csv')
        df = pd.DataFrame.from_dict(temp)
        df.to_csv(fname_lab_conv, index=False)
        self.fname_lab_conv = fname_lab_conv
