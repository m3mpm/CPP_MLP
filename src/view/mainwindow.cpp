#include "mainwindow.h"

#include "QtWidgets/qlabel.h"

namespace cpp_mlp {
MainWindow::MainWindow(Controller *ctrl, QWidget *parent)
    : QMainWindow(parent),
      ui_(new Ui::MainWindow),
      ctrl_(ctrl),
      learning_(new Learning(ctrl)),
      testing_(new Testing(ctrl)),
      validation_(new Validation(ctrl)) {
  ui_->setupUi(this);
  graphic_ = new Graphic(ui_->graphic_layout);

  SetConnections();

  ctrl_->InitNeuralNetwork(0, 2);
  ctrl_->SetStatisticFunc(
      std::bind(&MainWindow::signalMetrics, this, std::placeholders::_1));
}

void MainWindow::SetConnections() {
  connect(ui_->pushButton_recognize, &QPushButton::clicked, ui_->Canvas,
          &Painter::slotRecognizeFromCanvas);
  connect(ui_->Canvas, &Painter::signalSendDataToRecognize, this,
          &MainWindow::getDataToRecognize);

  connect(this, &MainWindow::signalMetrics, graphic_,
          &Graphic::addCurrentInGraphicSeries);
  connect(this, &MainWindow::signalMetrics, this,
          &MainWindow::setCurrentValues);

  connect(validation_, &Validation::signalSetGraphRange, graphic_,
          &Graphic::setRange);
  connect(learning_, &Learning::signalSetGraphRange, graphic_,
          &Graphic::setRange);
  connect(testing_, &Testing::signalSetGraphRange, graphic_,
          &Graphic::setRange);
}

MainWindow::~MainWindow() {
  ctrl_->StopWork();
  delete ui_;
  delete learning_;
  delete graphic_;
  delete testing_;
}

void MainWindow::on_pushButton_learn_clicked() { ShowDialog(learning_); }

void MainWindow::on_pushButton_experiment_clicked() { ShowDialog(testing_); }

void MainWindow::on_pushButton_cross_valid_clicked() {
  ui_->tabWidget->setCurrentIndex(1);
  ShowDialog(validation_);
}

void MainWindow::ShowDialog(QDialog *window) {
  ui_->tabWidget->setCurrentIndex(1);
  window->setModal(true);
  window->exec();
}

void MainWindow::on_pushButton_save_weigths_clicked() {
  QString filename = QFileDialog::getSaveFileName(
      this, tr("Save weigths"), QDir::homePath(), "Csv File (*.csv)");
  ctrl_->SaveWeight(filename.toStdString().c_str());
}

void MainWindow::getDataToRecognize(std::vector<double> pic_data) {
  if (IsEmptyData(pic_data)) {
    QMessageBox::critical(this, "critical", "Warning! Picture is clear!");
  } else {
    size_t answer = ctrl_->GetResult(pic_data);
    ui_->Canvas->update();
    ui_->showAnswer->setText((QString)((char)(answer + 65)));
  }
}

void MainWindow::on_pushButton_loadPicture_clicked() {
  QString start_path = src_dir + "/resources/data-samples/";
  QString file = QFileDialog::getOpenFileName(this, "Открыть файл", start_path,
                                              "Bmp File (*.bmp)");

  if (!file.isEmpty()) {
    QImage image(file, "BMP");
    ui_->Canvas->GetLinkToImage() =
        image.scaled(512, 512).convertToFormat(QImage::Format_ARGB32);
    ui_->pushButton_recognize->click();
  } else {
    ui_->Canvas->ClearDrawingArea();
  }
}

void MainWindow::on_pushButton_load_weights_clicked() {
  if (ui_->checkBox_manual_load_weights->isChecked())
    SetWeights();
  else
    SetWeights(ui_->comboBox_hidden_layers_amount->currentText().toInt());
}

void MainWindow::SetWeights() {
  QString start_path = src_dir + "resources/weigths/";
  QString filepath =
      QFileInfo(QFileDialog::getOpenFileName(this, "Открыть файл", start_path,
                                             "Csv File (*.csv)"))
          .absoluteFilePath();

  if (!filepath.isEmpty() && IsCorrectFile(filepath)) {
    ctrl_->SetWeights(filepath.toStdString());
    SetFilenameToLabel(ui_->label_load_weights_filename_status, filepath, "");
  } else {
    SetFilenameToLabel(ui_->label_load_weights_filename_status, filepath,
                       "Некорректный файл с весами!");
    QMessageBox::critical(this, "critical",
                          "Выбран некорректный файл с весами!");
  }
}

void MainWindow::SetWeights(size_t num_hidden_layers_) {
  QString path = QFileInfo(src_dir + "resources/weigths/").absoluteFilePath();
  path =
      path + "/weigths_" + QString::number(num_hidden_layers_) + "_layers.csv";

  if (!path.isEmpty() && IsCorrectFile(path)) {
    ctrl_->SetWeights(path.toStdString());
    SetFilenameToLabel(ui_->label_load_weights_filename_status, path, "");
  } else {
    SetFilenameToLabel(ui_->label_load_weights_filename_status, path,
                       "Некорректный файл с весами!");
    QMessageBox::critical(this, "critical",
                          "Выбран некорректный файл с весами!");
  }
}

void MainWindow::on_pushButton_load_dataset_clicked() {
  QString dataset_path;
  if (ui_->checkBox_manual_load_dataset->isChecked()) {
    QString start_path = src_dir + "resources/datasets/";
    dataset_path =
        QFileInfo(QFileDialog::getOpenFileName(this, "Открыть файл", start_path,
                                               "Csv File (*.csv)"))
            .absoluteFilePath();

    if (!dataset_path.isEmpty()) {
      ctrl_->SetDataset(dataset_path.toStdString());
      SetFilenameToLabel(ui_->label_load_dateset_filename_status, dataset_path,
                         ", файл загружен");
      ChangeEnableButtons(true);
    } else {
      SetFilenameToLabel(ui_->label_load_dateset_filename_status, dataset_path,
                         "Некорректный dataset-файл!");
      ChangeEnableButtons(false);
      QMessageBox::critical(this, "critical",
                            "Выбран некорректный dataset-файл!");
    }
  } else {
    if (ui_->comboBox_dataset_list->currentIndex() == 0) {
      dataset_path = src_dir + "resources/datasets/emnist-letters-train.csv";
    } else {
      dataset_path = src_dir + "resources/datasets/emnist-letters-test.csv";
    }
    ctrl_->SetDataset(dataset_path.toStdString());
    SetFilenameToLabel(ui_->label_load_dateset_filename_status, dataset_path,
                       ", файл загружен");
    ChangeEnableButtons(true);
  }
}

void MainWindow::setCurrentValues(Metrics currentValues) {
  ui_->averageAccuracyValue->setText(QString::number(currentValues.accuracy));
  ui_->precisionValue->setText(QString::number(currentValues.precision));
  ui_->recallValue->setText(QString::number(currentValues.recall));
  ui_->fMeasureValue->setText(QString::number(currentValues.f_measure));
  ui_->totalElapsedTimeValue->setText(
      QString::number(currentValues.total_time));
}

void MainWindow::InitNeuralNetwork() {
  ctrl_->InitNeuralNetwork(
      ui_->comboBox_network_type->currentIndex(),
      ui_->comboBox_hidden_layers_amount->currentText().toInt());
  ResetLoadLabels();
}

void MainWindow::SetFilenameToLabel(QLabel *label, QString file, QString msg) {
  QString file_name = getLastWord(file);
  if (msg.isEmpty()) {
    label->setText(file_name);
  } else {
    label->setText(file_name + " " + msg);
  }
}

bool MainWindow::IsEmptyData(std::vector<double> &data) {
  return !std::count_if(data.begin(), data.end(), [](double value) {
    return value > std::numeric_limits<double>::epsilon();
  });
}

bool MainWindow::IsCorrectFile(QString path) {
  bool result = false;
  size_t count = CountLinesInFile(path.toStdString());
  size_t num_hidden_layers =
      ui_->comboBox_hidden_layers_amount->currentText().toInt();
  if (num_hidden_layers == 2) {
    if (count == 314) result = true;
  } else if (num_hidden_layers == 3) {
    if (count == 458) result = true;
  } else if (num_hidden_layers == 4) {
    if (count == 602) result = true;
  } else if (num_hidden_layers == 5) {
    if (count == 746) result = true;
  }
  return result;
}

size_t MainWindow::CountLinesInFile(std::string path) {
  size_t count = 0;
  std::string line;
  std::fstream file(path, std::fstream::in);
  if (file.is_open()) {
    while (std::getline(file, line)) count++;
  }
  return count;
}

void MainWindow::on_comboBox_network_type_currentIndexChanged(int) {
  InitNeuralNetwork();
}

void MainWindow::on_comboBox_hidden_layers_amount_currentIndexChanged(int) {
  InitNeuralNetwork();
}

void MainWindow::ResetLoadLabels() {
  ui_->label_load_weights_filename_status->setText("Файл не загружен");
}

void MainWindow::ChangeEnableButtons(bool state) {
  ui_->pushButton_learn->setEnabled(state);
  ui_->pushButton_experiment->setEnabled(state);
  ui_->pushButton_cross_valid->setEnabled(state);
}

}  // namespace cpp_mlp
