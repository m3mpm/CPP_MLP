#include "testing.h"

#include "ui_testing.h"

namespace cpp_mlp {
Testing::Testing(Controller *ctrl, QWidget *parent)
    : QDialog(parent), ui_(new Ui::Testing), ctrl_{ctrl} {
  ui_->setupUi(this);

  connect(this, &Testing::signalProgress, ui_->progressBar,
          &QProgressBar::setValue);
  connect(ui_->pushButton_stop_test, &QPushButton::clicked, this,
          &Testing::reset);
}

Testing::~Testing() { delete ui_; }

void Testing::closeEvent(QCloseEvent *) { reset(); }

void Testing::on_pushButton_start_test_clicked() {
  const double rate = ui_->spinBox_rate->value();
  if (rate != 0) {
    emit signalSetGraphRange(1);
    ctrl_->SetProgressFunc(
        std::bind(&Testing::signalProgress, this, std::placeholders::_1));

    std::thread trd([this, rate]() {
      ui_->pushButton_start_test->setEnabled(false);
      ctrl_->Experiment(rate);
      ui_->pushButton_start_test->setEnabled(true);
    });
    trd.detach();
  } else
    ui_->progressBar->setValue(100);
}

void Testing::reset() {
  ctrl_->StopWork();
  ui_->pushButton_start_test->setEnabled(true);
  ui_->progressBar->reset();
}

}  // namespace cpp_mlp
