#include "learning.h"

#include "ui_learning.h"

namespace cpp_mlp {
Learning::Learning(Controller *ctrl, QWidget *parent)
    : QDialog(parent), ui_(new Ui::learning), ctrl_{ctrl} {
  ui_->setupUi(this);

  connect(this, &Learning::signalProgress, ui_->progressBar,
          &QProgressBar::setValue);
  connect(ui_->pushButton_stop_learn, &QPushButton::clicked, this,
          &Learning::reset);
}

Learning::~Learning() { delete ui_; }

void Learning::closeEvent(QCloseEvent *) { reset(); }

void Learning::on_pushButton_start_learn_clicked() {
  auto progress_ptr =
      std::bind(&Learning::signalProgress, this, std::placeholders::_1);

  ctrl_->SetProgressFunc(progress_ptr);
  emit signalSetGraphRange(ui_->spinBox->text().toInt());

  std::thread trd([this]() {
    ui_->pushButton_start_learn->setEnabled(false);
    ctrl_->Train(ui_->spinBox->text().toInt());
    ui_->pushButton_start_learn->setEnabled(true);
  });
  trd.detach();
}

void Learning::reset() {
  ctrl_->StopWork();
  ui_->progressBar->reset();
  ui_->pushButton_start_learn->setEnabled(true);
}

}  // namespace cpp_mlp
