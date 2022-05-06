"""The main Graphical User interface script.
This is the script that should be executed.
"""
import configparser
import os
import sys
import time

from itertools import repeat

from osgeo import ogr

from PySide2 import QtWidgets
from PySide2.QtWidgets import QApplication, QFileDialog, QMessageBox
from PySide2.QtGui import QIntValidator, QIcon

from main import main as predmain
from main import make_iterables, multiple_realizations
from uis.MainWindow import Ui_MainWindow


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    """Main gui window class

    """

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)
        self.child_win = None
        # the current icon comes from https://uxwing.com/
        # that has a permissive license
        self.setWindowIcon(QIcon(os.path.normpath('resources/icon.png')))
        self.setWindowTitle('Predictive Mapping')

        #############################################################################
        # validations
        self.only_pos = QIntValidator()
        self.only_pos.setRange(1, 999999999)
        
        for line_edit in [self.lineEdit_atLeast, 
                          self.lineEdit_maxSamples, 
                          self.lineEdit_seed]:
            line_edit.setValidator(self.only_pos)
        
        #############################################################################
        # connections
        self.pushButton_inputFileLito.clicked.connect(self.on_input_lito)
        self.pushButton_inputFilesFeatures.clicked.connect(
            self.on_input_features)
        self.pushButton_inputFileLimit.clicked.connect(self.on_input_limit)
        self.pushButton_outputDir.clicked.connect(self.on_output_dir)

        self.pushButton_OK.clicked.connect(self.on_ok)

        self.checkBox_cartesianProd.setEnabled(False)
        self.checkBox_coords.clicked.connect(self.on_coords_click)

    def on_input_lito(self):
        """Checks the input litology file
        """
        ffilter = "ESRI Shapefile(*.shp);;Geopackage (*.gpkg);; All files (*.*)"
        fname, _ = QFileDialog.getOpenFileName(self,
                                               'Select the target file',
                                               filter=ffilter
                                               )
        
        fname = os.path.normpath(fname)

        input_lito = ogr.Open(fname)

        if input_lito is None:
            button = QMessageBox.warning(self,
                                        "Predictive mapping",
                                        f"Please make sure {fname}" \
                                            + "is a valid and unlocked vector file.")

        else:
            self.lineEdit_inputFileLito.setText(fname)

            # check geometry:
            layer = input_lito.GetLayer()
            layer_defn = layer.GetLayerDefn()
            geom_type = ogr.GeometryTypeToName(layer_defn.GetGeomType())

            # update the minimum number of samples per class according to 
            # vector type
            if "point" in geom_type.lower():
                self.lineEdit_atLeast.setText('5')
            if "polygon" in geom_type.lower():
                self.lineEdit_atLeast.setText('40')

            # populate the combobox so user selects the field to be mapped:
            self.comboBox_fieldName.clear()
            self.comboBox_fieldName.addItems([field.name for field in layer.schema])

            # try to find SIGLA_UNID:
            target_field_idx = self.comboBox_fieldName.findText('SIGLA_UNID')
            self.comboBox_fieldName.setCurrentIndex(target_field_idx)

            # populate the combobox so user selects the ID field:
            self.comboBox_id.clear()
            self.comboBox_id.addItems([field.name for field in layer.schema])

            # try to find OBJECTID:
            target_field_idx = self.comboBox_id.findText('OBJECTID')
            self.comboBox_id.setCurrentIndex(target_field_idx)

        # close the file:
        input_lito = None


    def on_input_features(self):
        """Checks the feature files
        """
        ffilter = "Geotiff (*.tif);;ArcView binary raster grid (*.flt);; All files (*.*)"
        fnames, _ = QFileDialog.getOpenFileNames(self,
                                                 'Select the feature raster files',
                                                 filter=ffilter
                                                 )
        fnames = [os.path.normpath(fname) for fname in fnames]

        self.plainTextEdit_features.setPlainText('\n'.join(fnames))

    def on_input_limit(self):
        """Checks the input limit file
        """
        ffilter = "ESRI Shapefile(*.shp);;Geopackage (*.gpkg);; All files (*.*)"
        fname, _ = QFileDialog.getOpenFileName(self,
                                               'Select the bounding region file',
                                               filter=ffilter
                                               )

        self.lineEdit_inputFileLimit.setText(os.path.normpath(fname))

    def on_output_dir(self):
        """Checks the output directory
        """

        dname = QFileDialog.getExistingDirectory(self,
                                                 'Select the output directory',
                                                 )

        self.lineEdit_outputDir.setText(os.path.normpath(dname))

    def on_coords_click(self):
        """Checks if self.checkBox_coords is selected to enable
        checkBox_cartesianProd.
        """
        if self.checkBox_coords.isChecked():
            self.checkBox_cartesianProd.setEnabled(True)
        else:
            self.checkBox_cartesianProd.setChecked(False)
            self.checkBox_cartesianProd.setEnabled(False)

    def on_ok(self):
        """Start the program with the selected files
        """
        # set up config file:
        fnames_features = self.plainTextEdit_features.toPlainText()
        fname_target = self.lineEdit_inputFileLito.text()
        fname_limit = self.lineEdit_inputFileLimit.text()
        dir_out = self.lineEdit_outputDir.text()
        
        target_field = self.comboBox_fieldName.currentText()
        object_id = self.comboBox_id.currentText()

        discard_less_than = int(self.lineEdit_atLeast.text())
        max_samples_per_class = int(self.lineEdit_maxSamples.text())

        use_coords = self.checkBox_coords.isChecked()
        use_cartesian_prod = self.checkBox_cartesianProd.isChecked()
        run_pca = self.checkBox_PCA.isChecked()
        pca_percent = self.comboBox_PCAPercent.currentText()

        rand_seed_num = int(self.lineEdit_seed.text())
        number_of_realizations = int(self.comboBox_numberOfRealizations.currentText())

        config = configparser.ConfigParser()
        config['io'] = {'fnames_features': fnames_features,
                        'fname_target': fname_target,
                        'fname_limit': fname_limit,
                        'dir_out': dir_out}

        config['options'] = {'target_field': target_field,
                             'object_id': object_id,
                             'discard_less_than': discard_less_than,
                             'max_samples_per_class': max_samples_per_class, 
                             'use_coords': use_coords,
                             'use_cartesian_prod': use_cartesian_prod,
                             'run_pca': run_pca, 
                             'pca_percent': pca_percent}
        
        config['advanced'] = {'rand_seed_num': rand_seed_num, 
                              'number_of_realizations': number_of_realizations}

        # Assume program can be executed
        is_runnable = True

        for (key, val) in config.items('io'):

            if key == 'dir_out'   :
                if not os.path.isdir(val):
                    button = QMessageBox.question(self, 
                             "Predictive Mapping", 
                             f"{val} does not exist. Do you want to create it?")
                    if button == QMessageBox.No:
                        is_runnable = False

            elif key == 'fnames_features':
                for fname in val.split('\n'):
                    if not os.path.isfile(fname):
                        button = QMessageBox.warning(self,
                                "Predictive mapping",
                                f"Please make sure {fname} is a valid raster file.")
                        is_runnable = False
            else:
                if not os.path.isfile(val):
                    button = QMessageBox.warning(self,
                            "Predictive mapping",
                            f"Please make sure {fname} is a valid vector file.")
                    is_runnable = False

        if  discard_less_than < 5:
            button = QMessageBox.warning(self,
                    "Predictive mapping",
                    "Oops! SMOTE needs at least 5 samples per class. \n" \
                      +f"You selected ({discard_less_than}) samples.")
            is_runnable = False

        if  discard_less_than >= max_samples_per_class:
            button = QMessageBox.warning(self,
                    "Predictive mapping",
                    "Please make sure the Minimum number of samples per class " \
                      +f"({discard_less_than}) is smaller than the " \
                      +f"Maximum number of samples per class ({max_samples_per_class})")
            is_runnable = False

        if is_runnable:
            
            if number_of_realizations > 1:
                print(f'Program will compute {number_of_realizations} realizations.')

                # update the random seeds:
                rand_seed_num = range(rand_seed_num, rand_seed_num+number_of_realizations)
                config['advanced']['rand_seed_num'] = '\n'.join(str(rs) for rs in rand_seed_num)

                # write config file
                with open(os.path.join(dir_out, 'config.ini'), 'w', encoding='utf-8') as configfile:
                    config.write(configfile)

                # convert string to text
                fnames_features = fnames_features.split('\n') 

                # update the internal dir_out folders:
                dir_out = [os.path.join(dir_out, f'r{realization}') for realization in rand_seed_num]

                # create iterable arguments:
                iter_kwargs=make_iterables(fnames_features=fnames_features, 
                                           fname_target=fname_target, 
                                           fname_limit=fname_limit, 
                                           dir_out=dir_out,
                                           target_field=target_field,
                                           object_id=object_id,
                                           discard_less_than=discard_less_than, 
                                           max_samples_per_class=max_samples_per_class, 
                                           use_coords=use_coords,
                                           use_cartesian_prod=use_cartesian_prod,
                                           run_pca=run_pca, 
                                           pca_percent=pca_percent,
                                           rand_seed_num=rand_seed_num)
                
                # fnames_features is iterable, but not set up as expected:
                iter_kwargs['fnames_features'] = repeat(iter_kwargs['fnames_features'])

                # call threaded function:
                start_time = time.perf_counter()
                multiple_realizations(**iter_kwargs)

                end_time = time.perf_counter()

                print(f'\nExecution time: {(end_time-start_time)/60:.2f} minutes')

            else:
                # write config file:
                with open('config.ini', 'w', encoding='utf-8') as configfile:
                    config.write(configfile)

                # call the main function:
                start_time = time.perf_counter()
                predmain(config['io']['fnames_features'].split('\n'),
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

                # save the configuration file for reference:
                os.replace('config.ini', 
                        os.path.join(dir_out, 'config.ini'))

                end_time = time.perf_counter()

                print(f'\nExecution time: {(end_time-start_time)/60:.2f} minutes')


def main():
    """Main function
    """
    app = QApplication(sys.argv)

    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
