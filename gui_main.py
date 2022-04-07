"""The main Graphical User interface script.
This is the script that should be executed.
"""
import configparser
import os
import sys
import time

from osgeo import ogr

from PySide2 import QtWidgets
from PySide2.QtWidgets import QApplication, QFileDialog, QMessageBox
from PySide2.QtGui import QIntValidator

from main import main as predmain
from uis.MainWindow import Ui_MainWindow


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    """Main gui window class

    """

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)
        self.child_win = None
        self.display_wins = {}

        #############################################################################
        # validations
        self.only_pos = QIntValidator()
        self.only_pos.setRange(1, 9999999)
        
        for line_edit in [self.lineEdit_atLeast, 
                          self.lineEdit_maxSamples]:
            line_edit.setValidator(self.only_pos)
        
        #############################################################################
        # connections
        self.pushButton_inputFileLito.clicked.connect(self.on_input_lito)
        self.pushButton_inputFilesFeatures.clicked.connect(
            self.on_input_features)
        self.pushButton_inputFileLimit.clicked.connect(self.on_input_limit)
        self.pushButton_outputDir.clicked.connect(self.on_output_dir)

        self.pushButton_OK.clicked.connect(self.on_ok)

    def on_input_lito(self):
        """Checks the input litology file
        """
        ffilter = "ESRI Shapefile(*.shp);;Geopackage (*.gpkg);; All files (*.*)"
        fname, _ = QFileDialog.getOpenFileName(self,
                                               'Selecione o arquivo de litologia',
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

    def on_input_features(self):
        """Checks the feature files
        """
        ffilter = "Geotiff (*.tif);;ArcView binary raster grid (*.flt);; All files (*.*)"
        fnames, _ = QFileDialog.getOpenFileNames(self,
                                                 'Selecione os arquivo rasters de entrada',
                                                 filter=ffilter
                                                 )
        fnames = [os.path.normpath(fname) for fname in fnames]

        self.plainTextEdit_features.setPlainText('\n'.join(fnames))

    def on_input_limit(self):
        """Checks the input limit file
        """
        ffilter = "ESRI Shapefile(*.shp);;Geopackage (*.gpkg);; All files (*.*)"
        fname, _ = QFileDialog.getOpenFileName(self,
                                               'Selecione o arquivo que delimita a area',
                                               filter=ffilter
                                               )

        self.lineEdit_inputFileLimit.setText(os.path.normpath(fname))

    def on_output_dir(self):
        """Checks the output directory
        """

        dname = QFileDialog.getExistingDirectory(self,
                                                 'Select output directory',
                                                 )

        self.lineEdit_outputDir.setText(os.path.normpath(dname))

    def on_ok(self):
        """Start the program with the selected files
        """
        # set up config file:
        fnames_features = self.plainTextEdit_features.toPlainText()
        fname_target = self.lineEdit_inputFileLito.text()
        fname_limit = self.lineEdit_inputFileLimit.text()
        dir_out = self.lineEdit_outputDir.text()
        
        discard_less_than = int(self.lineEdit_atLeast.text())
        max_samples_per_class = int(self.lineEdit_maxSamples.text())
        
        config = configparser.ConfigParser()
        config['io'] = {'fnames_features': fnames_features,
                        'fname_target': fname_target,
                        'fname_limit': fname_limit,
                        'dir_out': dir_out}

        config['options'] = {'discard_less_than': discard_less_than,
                             'max_samples_per_class': max_samples_per_class}

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
            # write config file:
            with open('config.ini', 'w', encoding='utf-8') as configfile:
                config.write(configfile)

            # call the main function:
            start_time = time.perf_counter()
            predmain(config['io']['fnames_features'].split('\n'),
                    config['io']['fname_target'],
                    config['io']['fname_limit'],
                    config['io']['dir_out'], 
                    config['options']['discard_less_than'], 
                    config['options']['max_samples_per_class'] )

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
