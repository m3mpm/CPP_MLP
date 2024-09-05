#ifndef GRAPHIC_H
#define GRAPHIC_H

#include <QBoxLayout>
#include <QMainWindow>
#include <QtCharts/QBarCategoryAxis>
#include <QtCharts/QBarSeries>
#include <QtCharts/QBarSet>
#include <QtCharts/QCategoryAxis>
#include <QtCharts/QChartView>
#include <QtCharts/QHorizontalStackedBarSeries>
#include <QtCharts/QLegend>
#include <QtCharts/QLineSeries>
#include <QtCharts/QPieSeries>
#include <QtCharts/QPieSlice>
#include <QtWidgets/QApplication>
#include <QtWidgets/QMainWindow>
#include <tuple>

#include "model/metrics.h"
#include "ui_mainwindow.h"

namespace cpp_mlp {
class Graphic : public QWidget {
  Q_OBJECT

 public:
  explicit Graphic(QBoxLayout *attachmentPlace);
  Metrics GetCurrentValues();
  void Clear();

 public slots:
  void addCurrentInGraphicSeries(cpp_mlp::Metrics);
  void setRange(size_t range);

 private:
  size_t epoch_amount_ = 0;
  size_t epoch_step_ = 0;
  double time_ = 0;

  QLineSeries *average_accuracy, *precision, *recall, *f_measure;

  QChart *chart;
  QChartView *chartView;

  void Draw();
};
}  // namespace cpp_mlp

#endif  // GRAPHIC_H
