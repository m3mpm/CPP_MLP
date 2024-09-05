#ifndef PAINTER_H
#define PAINTER_H

#include <QConicalGradient>
#include <QDir>
#include <QImage>
#include <QLinearGradient>
#include <QMouseEvent>
#include <QPainter>
#include <QPushButton>
#include <QRadialGradient>
#include <QWidget>
#include <fstream>
#include <iostream>

class Painter : public QWidget {
  Q_OBJECT

 public:
  explicit Painter(QWidget *parent = nullptr);
  ~Painter(){};

  QImage &GetLinkToImage() { return image_; }
  void ClearDrawingArea();

 public slots:
  void slotRecognizeFromCanvas();

 private:
  QImage image_;
  std::vector<double> data_{};
  Qt::MouseButton button_;
  QPoint start_;

  int kSIZE = 28;

  void mousePressEvent(QMouseEvent *event) override;
  void mouseMoveEvent(QMouseEvent *event) override;
  void paintEvent(QPaintEvent *event) override;

  void ConvertPictureToArray();
  void Draw(const QPoint &);

 signals:
  void signalSendDataToRecognize(std::vector<double>);
};

#endif  // PAINTER_H
