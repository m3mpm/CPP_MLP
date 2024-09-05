#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QDir>
#include <QFileDialog>
#include <QImage>
#include <QMainWindow>
#include <QMessageBox>
#include <QProgressDialog>
#include <QStandardPaths>
#include <QTemporaryDir>
#include <QThread>
#include <QtWidgets/QMainWindow>
#include <functional>

#include "../controller/controller.h"
#include "graphic.h"
#include "learning.h"
#include "model/metrics.h"
#include "painter.h"
#include "testing.h"
#include "ui_mainwindow.h"
#include "validation.h"

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

namespace cpp_mlp {
class MainWindow : public QMainWindow {
  Q_OBJECT

 public:
  explicit MainWindow(Controller *ctrl, QWidget *parent = nullptr);
  ~MainWindow();

 private:
  Ui::MainWindow *ui_;

  Validation *validation_;
  Learning *learning_;
  Controller *ctrl_;
  Graphic *graphic_;
  Testing *testing_;
  QString src_dir = QCoreApplication::applicationDirPath() + "/../../../";

  void SetFilenameToLabel(QLabel *label, QString file, QString msg);
  QString getLastWord(QString words) { return words.split("/").last(); }
  bool IsEmptyData(std::vector<double> &);
  size_t CountLinesInFile(std::string);
  bool IsCorrectFile(QString);
  void InitNeuralNetwork();
  void SetWeights(size_t);
  void ResetLoadLabels();
  void SetConnections();
  void SetWeights();
  void ShowDialog(QDialog *);
  void ChangeEnableButtons(bool state);

 private slots:
  void on_comboBox_hidden_layers_amount_currentIndexChanged(int);
  void on_comboBox_network_type_currentIndexChanged(int);
  void on_pushButton_load_dataset_clicked();
  void on_pushButton_load_weights_clicked();
  void on_pushButton_save_weigths_clicked();
  void on_pushButton_cross_valid_clicked();
  void on_pushButton_loadPicture_clicked();
  void on_pushButton_experiment_clicked();
  void on_pushButton_learn_clicked();

  void setCurrentValues(cpp_mlp::Metrics);
  void getDataToRecognize(std::vector<double>);

 signals:
  void signalRecognizeImportImage();
  void signalMetrics(cpp_mlp::Metrics);
};
}  // namespace cpp_mlp

#endif  // MAINWINDOW_H
