#ifndef TESTING_H
#define TESTING_H

#include <QDialog>
#include <QFileDialog>
#include <QMessageBox>
#include <QTimer>
#include <mutex>
#include <thread>

#include "../controller/controller.h"
#include "../model/metrics.h"
#include "QtWidgets/qstatusbar.h"

namespace Ui {
class Testing;
}

namespace cpp_mlp {
class Testing : public QDialog {
  Q_OBJECT

 public:
  explicit Testing(Controller *ctrl, QWidget *parent = nullptr);
  ~Testing();

 private:
  Ui::Testing *ui_;
  Controller *ctrl_;

  void closeEvent(QCloseEvent *);
  void SetFileName(QString);

 private slots:
  void on_pushButton_start_test_clicked();
  void reset();

 signals:
  void signalSetGraphRange(int);
  void signalProgress(int);
};

}  // namespace cpp_mlp

#endif  // TESTING_H
