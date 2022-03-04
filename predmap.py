"""
Main class for the predictive geological mapping
"""
import glob
import itertools
import os
import warnings  # desabilitar avisos
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from osgeo import gdal, osr
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     StratifiedKFold)
# pre-processing
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


class PredMap():
    """Main class
    """

    def __init__(self,
                 fnames_features,
                 fname_target,
                 fname_lab_conv,
                 fname_limit,
                 dir_out):
        """[summary]

        Args:
            fnames_features (list): list of features filenames (rasters)
            fname_target (os.path - file): filename of the target (polygon vector layer)
            fname_lab_conv (os.path - file): filename of the file mapping target labels to integer values
            fname_limit (os.path - file): filename of the limiting boundary (polygon vector layer)
            dir_out (os.path - directory): directory where the output files will be saved
        """
        self.fnames_features = fnames_features
        self.fname_target = fname_target
        self.fname_lab_conv = fname_lab_conv
        self.fname_limit = fname_limit
        self.dir_out = dir_out

        # integer value to be used as nan
        self.nanval = -9999

        # target attribute:
        self.target_attribute = 'SIGLA_UNID'
        # integer identifier
        self.object_id = 'OBJECTID'

        # this is not generic enough, but better to be declared here
        # than in the middle of the class
        self.list_of_features = ['ThU', 'Kperc', 'SRTM', 'GT', 'eU', 'eTh', 'ThK',
                                 'UK', 'CT', 'B02', 'B03', 'B04', 'B06', 'B07']

        # these will be assembled by the class
        self.X = None
        self.y = None
        self.y_pred = None
        self.target_raster = None
        self.le = None
        self.le_df = None
        self.lab_to_int = None
        self.int_to_lab = None
        self.target_raster_fname = None

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

        # label encoding for the model:
        self.le = LabelEncoder()
        self.le.fit(out.ravel()[out.ravel() != self.nanval])

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
        """Transform raster to numpy array to be used for training and eval
        """
        feats = []
        for raster in self.feature_rasters:
            for idx in range(raster.RasterCount):
                # Read the raster band as separate variable
                band = raster.GetRasterBand(idx+1)
                # get numpy vals
                band_np = band.ReadAsArray()
                # assure band_np is float and not int to be able to asign np.nan
                # should we do this or define an "int nan"?
                band_np = band_np.astype(float)
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

    def prepare_to_fit(self):

        files = glob.glob(self.dir_out + '/*tif')

        aux = 0
        for file in files:

            ds = gdal.Open(file)

            gt = ds.GetGeoTransform()
            raster = ds.ReadAsArray()

            if aux == 0:
                df2 = pd.DataFrame.from_records(itertools.product(range(ds.RasterYSize), range(ds.RasterXSize)),
                                                columns=['Row', 'Column'])
                ds = None

            if 'SRTM'.lower() in file.lower():
                colname = 'SRTM'

                df2['SRTM'] = np.reshape(raster, (-1, 1))
            elif 'Landsat'.lower() in file.lower():
                # B02	B03	B04	B06	B07
                tmp = 0
                for colname in ['B02', 'B03', 'B04', 'B06', 'B07']:
                    rst = raster[tmp]
                    rst = np.nan_to_num(rst, nan=self.nanval)
                    df2[colname] = np.reshape(rst, (-1, 1))
                    tmp += 1
                    rst = None
            elif 'Litologia'.lower() in file.lower():
                colname = 'TARGET'
                y_temp = np.reshape(raster, (-1, 1))
                y_temp[y_temp != self.nanval] = self.le.transform(
                    y_temp[y_temp != self.nanval])
                df2[colname] = y_temp

            else:
                fname = Path(file).resolve().stem
                # find the index in the list of accepted features:
                try:
                    idx = [feat.lower() in fname.lower()
                           for feat in self.list_of_features].index(True)
                except ValueError:
                    print(
                        f'****Warning-Skipping unnacepted feature {fname}****')
                    pass
                else:
                    colname = self.list_of_features[idx]
                    df2[colname] = np.reshape(raster, (-1, 1))
            aux += 1
            ds = None

            print(file, raster[raster == self.nanval].sum()/self.nanval)

        print(df2.columns.values)

        return df2

    def fit(self):
        """Fit XGboost with grid search
        """
        print('VAR to copy')
        print(self.target_raster.GetGeoTransform())
        print(self.proj.ExportToWkt())

        from functions import (MaskedPCA, createPredTable,
                               customTrainTestSplit, validationReport)

        df_original = self.prepare_to_fit()
        df = self.prepare_to_fit()
        # drop all nan vals:

        nan_mask = df.isin([self.nanval]).any(axis=1)
        if self.nanval in self.le.classes_:
            nan_transf = self.le.transform([self.nanval]).item()
            nan_mask = nan_mask + df['TARGET'] == nan_transf

        print(f'Before dropping nan values: {df.shape}')
        df = df[~nan_mask]
        print(f'After dropping nan values: {df.shape}')
        self.nan_mask = nan_mask
        #  print(df.columns)

        lito_count = df.TARGET.value_counts() < 40
        litologias = lito_count.index
        aux = 0
        for l in lito_count.tolist():
            if l:
                print('Discard Litology: ', litologias[aux])
                df = df[df['TARGET'] != litologias[aux]]
            aux += 1
        FEAT = self.list_of_features
        COORD = ['Row', 'Column']

        X_train, y_train, coord_train, X_test, y_test, coord_test = customTrainTestSplit(df, FEAT, COORD,
                                                                                         samp_per_class=150,
                                                                                         threshold=0.7,
                                                                                         coords=True)

        print(
            'Treino -> features: {0}   |  target: {1}'.format(X_train.shape, y_train.shape))
        print(
            'Teste  -> features: {0} |  target: {1}'.format(X_test.shape, y_test.shape))

        # dataframe de treino
        train_loc = pd.DataFrame(coord_train, columns=COORD)
        train_feat = pd.DataFrame(X_train, columns=FEAT)
        train = pd.concat([train_loc, train_feat], axis=1)
        train['TARGET'] = y_train

        # dataframe de teste
        test_loc = pd.DataFrame(coord_test, columns=COORD)
        test_feat = pd.DataFrame(X_test, columns=FEAT)
        test = pd.concat([test_loc, test_feat], axis=1)
        test['TARGET'] = y_test

        std_scaler = StandardScaler()
        X_train_std = std_scaler.fit_transform(X_train)
        df_train_std = pd.DataFrame(X_train_std, columns=df[FEAT].columns)

        # PCAs
        pca = PCA(n_components=5)
        pcs = pca.fit_transform(X_train_std[:, np.arange(9, 14)])

        # dataframe com as bandas Landsat
        orig_bands = pd.DataFrame(X_train_std[:, np.arange(9, 14)], columns=[
                                  'B02', 'B03', 'B04', 'B06', 'B07'])
        # dataframe com as componentes principais
        principal_comps = pd.DataFrame(
            pcs, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5'])
        # Combinação dos dois dataframes
        combined_pca = pd.concat([orig_bands, principal_comps], axis=1)

        # correlation matrix of PCA
        corr_matrix = combined_pca.corr()
        corr_plot_data = corr_matrix[:-
                                     len(principal_comps.columns)].loc[:, 'PC1':]

        #  MaskedPCA to Landsat
        mask = np.arange(9, 14)
        masked_pca = MaskedPCA(n_components=1, mask=mask)

        X_train_pca = masked_pca.fit_transform(X_train_std)
        print(
            f'Dimensions of train features (before-PCA) = {X_train_std.shape}')
        print(
            f'Dimensions of train features (after-PCA) = {X_train_pca.shape}\n')

        PCA_FEAT = list(FEAT[:9]) + ['PC1']
        df_X_train_pca = pd.DataFrame(X_train_pca, columns=PCA_FEAT)

        pca_corr = df_X_train_pca.corr(method='pearson').round(2)

        # heatmap of linear correlation
        plt.figure(figsize=(7, 6))
        mask = np.triu(np.ones_like(pca_corr, dtype=np.bool))
        ax = sns.heatmap(
            pca_corr, annot=True,
            cmap='coolwarm', cbar=True,
            mask=mask, vmin=-1.0, vmax=1.0
        )
        ax.set_xticklabels(PCA_FEAT, rotation=45)
        ax.set_yticklabels(PCA_FEAT, rotation=0)

        plt.savefig(self.dir_out+"/correlation_features.png", dpi=300)
        # delete of feature CT
        df.drop(['CT'], axis=1, inplace=True)

        # SMOTE
        X_train_smt, y_train_smt = SMOTE().fit_resample(X_train_pca, y_train)
        train_smt = pd.DataFrame(X_train_smt, columns=PCA_FEAT)
        train_smt['TARGET'] = y_train_smt

        nb_eg = 150
        scaler = StandardScaler()
        # PCA
        mask = np.arange(8, 13)
        dim_reduction = MaskedPCA(n_components=1, mask=mask)
        oversamp = SMOTE(random_state=42)
        n_folds = 5

        # cross-validation
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        #  performance metric
        metric = 'f1_weighted'

        FEAT.remove('CT')

        X_train, y_train, X_test, y_test = customTrainTestSplit(df, FEAT, COORD,
                                                                samp_per_class=nb_eg,
                                                                threshold=0.7)

        print(f"TRAIN: X {X_train.shape}, y {y_train.shape}")
        print(f"TEST: X {X_test.shape}, y {y_test.shape}")

        # XGB
        xgb_pipe = Pipeline(steps=[('scaler', scaler),
                                   ('dim_reduction', dim_reduction),
                                   ('smote', oversamp),
                                   ('clf', XGBClassifier(eval_metric='mlogloss', verbosity=0,
                                                         random_state=42))])
        # pipe = {"RF": rf_pipe}
        pipe = {"XGB": xgb_pipe}

        # XGB
        xgb_param = [{'clf__eta': [0.01, 0.015, 0.025, 0.05, 0.1],
                      'clf__learning_rate': [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
                      'clf__gamma': [0.05, 0.1, 0.3, 0.4, 0.5, 0.7, 0.9, 1.0],
                      'clf__max_depth': [3, 5, 7, 9, 12, 15, 17, 25],
                      'clf__min_child_weight': [1, 3, 5, 7],
                      'clf__subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                      'clf__colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
                      'clf__reg_lambda': [10 ** i for i in range(-3, 4)],
                      'clf__alpha': [10 ** i for i in range(-3, 4)]}]

        param = [xgb_param]

        dic_param = {}
        for k, p in zip(pipe.keys(), param):
            dic_param[k] = p

        best_params = []

        # Grid Search
        for m in ['XGB']:
            # random search
            random = RandomizedSearchCV(pipe[m], param_distributions=dic_param[m], cv=cv,
                                        scoring=metric, n_iter=50, random_state=42)
            print(X_train.shape, y_train.shape)
            random.fit(X_train, y_train)
            best_params.append(random.best_params_)
            print("----")
            print(m)
            print("Best parameters:", random.best_params_)
            print('{0} = {1}'.format(metric, round(random.best_score_, 3)))

        print(best_params[0])

        xgb = Pipeline(steps=[('scaler', scaler),
                              ('dim_reduction', dim_reduction),
                              ('smote', oversamp),
                              ('clf', XGBClassifier(subsample=best_params[0]['clf__subsample'],
                                                    reg_lambda=best_params[0]['clf__reg_lambda'],
                                                    min_child_weight=best_params[0]['clf__min_child_weight'],
                                                    max_depth=best_params[0]['clf__max_depth'],
                                                    learning_rate=best_params[0]['clf__learning_rate'],
                                                    gamma=best_params[0]['clf__gamma'],
                                                    eta=best_params[0]['clf__eta'],
                                                    colsample_bytree=best_params[0]['clf__colsample_bytree'],
                                                    alpha=best_params[0]['clf__alpha'],
                                                    random_state=42))])

        tuned_models = {"XGB": xgb}

        val_report = validationReport(tuned_models, X_train, y_train, cv)
        print(val_report)
        val_report.to_csv(os.path.join(self.dir_out, 'validation_report.csv'))

        for k in tuned_models.keys():
            tuned_models[k].fit(X_train, y_train)

        ŷ_xgb_train = tuned_models['XGB'].predict(X_train)

        dic_ŷ_train = {'XGB': ŷ_xgb_train}

        ŷ_xgb_test = tuned_models['XGB'].predict(X_test)

        dic_ŷ_test = {'XGB': ŷ_xgb_test}

        pred_map = createPredTable(dic_ŷ_train, dic_ŷ_test, train, test)

        df_sorted = pred_map.sort_values(by=['Row', 'Column'],
                                         ascending=[True, True])

        arr = df_sorted['Litology'].to_numpy()

        ypred = np.pad(arr.astype(float), (0, self.target_raster.RasterXSize *
                       self.target_raster.RasterYSize - arr.size))
        self.y_pred = ypred.reshape(self.target_raster.RasterXSize,
                                    self.target_raster.RasterYSize)

        # use the df_original set in the beginning of the function:
        X = df_original[FEAT].to_numpy()
        self.y_pred = tuned_models['XGB'].predict_proba(X)

        # reasign nan values based on mask:
        self.y_pred[nan_mask] = self.nanval

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
        out = np.argmax(self.y_pred, axis=1)
        # convert prediction to project "label":
        out = self.le.inverse_transform(out).astype(np.int16)
        # reasign nan values based on mask:
        out[self.nan_mask] = self.nanval
        # reshape to raster dimensions
        out = np.reshape(out, (self.target_raster.RasterYSize,
                               self.target_raster.RasterXSize))

        band.WriteArray(out)

        # set the "No Data Value"
        band.SetNoDataValue(self.nanval)

        # close to write the raster
        dest = None

    def write_report(self):
        """Evaluate model and write the metrics
        (e.g., confusion matrix, classification report)
        """
        pass
