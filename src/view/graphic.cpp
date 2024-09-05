#include "graphic.h"

namespace cpp_mlp {
Graphic::Graphic(QBoxLayout* attachmentPlace) {
  average_accuracy = new QLineSeries();
  average_accuracy->setName("average_accuracy");

  precision = new QLineSeries();
  precision->setName("precision");

  recall = new QLineSeries();
  recall->setName("recall");

  f_measure = new QLineSeries();
  f_measure->setName("f_measure");

  chart = new QChart();
  chart->addSeries(average_accuracy);
  chart->addSeries(precision);
  chart->addSeries(recall);
  chart->addSeries(f_measure);

  chart->setAnimationOptions(QChart::AllAnimations);

  chartView = new QChartView(chart);
  chartView->setRenderHint(QPainter::Antialiasing);

  attachmentPlace->addWidget(chartView);

  chart->createDefaultAxes();

  chart->axes(Qt::Horizontal).at(0)->setRange(0, 1);
  chart->axes(Qt::Vertical).at(0)->setRange(0, 1);
  chartView->update();
}

void Graphic::addCurrentInGraphicSeries(Metrics metrics) {
  epoch_step_++;
  average_accuracy->append(epoch_step_, metrics.accuracy);
  precision->append(epoch_step_, metrics.precision);
  recall->append(epoch_step_, metrics.recall);
  f_measure->append(epoch_step_, metrics.f_measure);
  time_ = metrics.total_time;

  Draw();
}

void Graphic::Draw() {
  chart->setAnimationOptions(QChart::SeriesAnimations);
  chart->axes(Qt::Horizontal)
      .at(0)
      ->setRange(0, QVariant::fromValue(epoch_amount_));
  chartView->update();
}

void Graphic::Clear() {
  epoch_amount_ = 0;
  epoch_step_ = 0;

  average_accuracy->clear();
  precision->clear();
  recall->clear();
  f_measure->clear();

  average_accuracy->append(0, 0);
  precision->append(0, 0);
  recall->append(0, 0);
  f_measure->append(0, 0);
}

Metrics Graphic::GetCurrentValues() {
  return Metrics{average_accuracy->at(epoch_step_).y(),
                 precision->at(epoch_step_).y(), recall->at(epoch_step_).y(),
                 f_measure->at(epoch_step_).y(), time_};
}

void Graphic::setRange(size_t range) {
  Clear();
  epoch_amount_ = range;
}

}  // namespace cpp_mlp
