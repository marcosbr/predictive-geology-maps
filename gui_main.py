"""The main Graphical User interface script.
This is the script that should be executed.
"""
import configparser
import os
import sys
import time

from PySide2 import QtWidgets
from PySide2.QtWidgets import QApplication, QFileDialog

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

        # connections
        self.pushButton_inputFileLito.clicked.connect(self.on_input_lito)
        self.pushButton_inputFilesFeatures.clicked.connect(
            self.on_input_features)
        self.pushButton_inputFileLimit.clicked.connect(self.on_input_limit)
        self.pushButton_inputFileLabConv.clicked.connect(
            self.on_input_lab_conv)
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

        self.lineEdit_inputFileLito.setText(os.path.normpath(fname))

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

    def on_input_lab_conv(self):
        """Checks the SIGLA_UNID-integer conversion file
        """
        ffilter = "Comma Separated Values(*.csv);; All files (*.*)"
        fname, _ = QFileDialog.getOpenFileName(self,
                                               'Selecione o arquivo para conversao entre geologia e numero inteiro',
                                               filter=ffilter
                                               )

        self.lineEdit_inputFileLabConv.setText(os.path.normpath(fname))

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
        fname_lab_conv = self.lineEdit_inputFileLabConv.text()
        dir_out = self.lineEdit_outputDir.text()
        config = configparser.ConfigParser()
        config['io'] = {'fnames_features': fnames_features,
                        'fname_target': fname_target,
                        'fname_limit': fname_limit,
                        'fname_lab_conv': fname_lab_conv,
                        'dir_out': dir_out}

        for (key, val) in config.items('io'):
            if key == 'dir_out':
                if not os.path.isdir(val):
                    print("Please select appropriate output directory")
                    return
            elif key == 'fnames_features':
                for fname in val.split('\n'):
                    if not os.path.isfile(fname):
                        print(f"Please make sure {fname} is a valid file")
            else:
                if not os.path.isfile(val):
                    print(f"Please make sure {val} is a valid file")
                    return

        # write config file:
        with open('config.ini', 'w', encoding='utf-8') as configfile:
            config.write(configfile)

        # call the main function:
        start_time = time.perf_counter()
        predmain(config['io']['fnames_features'].split('\n'),
                 config['io']['fname_target'],
                 config['io']['fname_lab_conv'],
                 config['io']['fname_limit'],
                 config['io']['dir_out'])

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
