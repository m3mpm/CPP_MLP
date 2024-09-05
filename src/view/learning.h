#ifndef LEARNING_H
#define LEARNING_H

#include <QDialog>
#include <thread>

#include "../controller/controller.h"
#include "../model/metrics.h"
#include "QtWidgets/qstatusbar.h"

namespace Ui {
class learning;
}

namespace cpp_mlp {
class Learning : public QDialog {
  Q_OBJECT

 public:
  explicit Learning(Controller *ctrl, QWidget *parent = nullptr);
  ~Learning();

 private:
  Ui::learning *ui_;
  Controller *ctrl_;

  void closeEvent(QCloseEvent *);

 private slots:
  void on_pushButton_start_learn_clicked();
  void reset();

 signals:
  void signalSetGraphRange(int);
  void signalProgress(int);
};
}  // namespace cpp_mlp

#endif  // LEARNING_H
