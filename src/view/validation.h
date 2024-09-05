#ifndef VALIDATION_H
#define VALIDATION_H

#include <QCloseEvent>
#include <QDialog>
#include <QFileDialog>
#include <QMessageBox>
#include <mutex>
#include <thread>

#include "../controller/controller.h"
#include "QtWidgets/qstatusbar.h"

namespace Ui {
class Validation;
}

namespace cpp_mlp {
class Validation : public QDialog {
  Q_OBJECT

 public:
  explicit Validation(Controller *ctrl, QWidget *parent = nullptr);
  ~Validation();

 private:
  Ui::Validation *ui_;
  Controller *ctrl_;
  QStatusBar *bar_;

  size_t count_ = 2;
  int stage_number_ = 1;

  void closeEvent(QCloseEvent *);
  void UnblockButtons();
  void BlockButtons();
  void Reset();

 private slots:
  void on_pushButton_start_validation_clicked();
  void on_pushButton_stop_validation_clicked();
  void showStageNameAndNumber(int);
  void tryUnblockButtons(int);

 signals:
  void signalUpdateProgress(int);
  void signalSetGraphRange(int);
};

}  // namespace cpp_mlp

#endif  // VALIDATION_H
