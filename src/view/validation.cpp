#include "validation.h"

#include "ui_validation.h"

namespace cpp_mlp {

Validation::Validation(Controller *ctrl, QWidget *parent)
    : QDialog(parent), ui_(new Ui::Validation), ctrl_{ctrl} {
  ui_->setupUi(this);

  bar_ = new QStatusBar(this);
  ui_->statusLayout->addWidget(bar_);
  bar_->addWidget(ui_->label_stage_name);
  bar_->addWidget(ui_->label_status);
  bar_->addWidget(ui_->label_stage_number);

  connect(this, &::cpp_mlp::Validation::signalUpdateProgress, ui_->progressBar,
          &QProgressBar::setValue);
  connect(ui_->progressBar, &QProgressBar::valueChanged, this,
          &Validation::showStageNameAndNumber);
  connect(ui_->progressBar, &QProgressBar::valueChanged, this,
          &Validation::tryUnblockButtons);
}

Validation::~Validation() { delete ui_; }

void Validation::closeEvent(QCloseEvent *event) {
  ctrl_->StopWork();
  Reset();
  UnblockButtons();
}

void Validation::on_pushButton_start_validation_clicked() {
  Reset();
  BlockButtons();

  auto progress_ptr = std::bind(&::cpp_mlp::Validation::signalUpdateProgress, this,
                                std::placeholders::_1);
  ctrl_->SetProgressFunc(progress_ptr);
  emit signalSetGraphRange(ui_->spinBox_groupAmount->value());

  std::thread trd(
      [this]() { ctrl_->CrossValidation(ui_->spinBox_groupAmount->value()); });
  trd.detach();

  ui_->label_stage_name->setText("Обучение: ");
  ui_->label_stage_number->setText("этап №" + QString::number(stage_number_));
  ui_->label_status->setText("в процессе, ");
}

void Validation::on_pushButton_stop_validation_clicked() {
  ctrl_->StopWork();
  Reset();
  UnblockButtons();
  ui_->label_status->setText("Процесс прерван");
}

void Validation::showStageNameAndNumber(int value) {
  if (value == 100) {
    count_++;
    if (count_ == ((ui_->spinBox_groupAmount->value() * 3) - 1)) {
      ui_->label_stage_name->setText("Тестирование: ");
    } else if (count_ % 3 == 0) {
      ui_->label_stage_name->setText("Тестирование: ");
      stage_number_++;
    } else {
      ui_->label_stage_name->setText("Обучение: ");
      ui_->label_stage_number->setText("этап №" +
                                       QString::number(stage_number_));
    }
  }
}

void Validation::tryUnblockButtons(int value) {
  if (value == 100) {
    if (count_ == ui_->spinBox_groupAmount->value() * 3) {
      UnblockButtons();
      ui_->label_status->setText("Завершено");
    }
  }
}

void Validation::BlockButtons() {
  ui_->pushButton_start_validation->setEnabled(false);
}

void Validation::UnblockButtons() {
  ui_->pushButton_start_validation->setEnabled(true);
}

void Validation::Reset() {
  count_ = 2;
  stage_number_ = 1;
  ui_->progressBar->setValue(0);
  ui_->label_stage_name->clear();
  ui_->label_stage_number->clear();
  ui_->label_status->clear();
}

}  // namespace cpp_mlp
