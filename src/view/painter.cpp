#include "painter.h"

#include <iostream>
#include <thread>

Painter::Painter(QWidget *parent)
    : QWidget{parent}, image_(QImage(QSize(560, 560), QImage::Format_RGB32)) {
  image_.fill(Qt::white);
}

void Painter::slotRecognizeFromCanvas() { ConvertPictureToArray(); }

void Painter::paintEvent(QPaintEvent *event) {
  Q_UNUSED(event);
  QPainter painter(this);
  QRect rec = event->rect();
  painter.drawPixmap(rec, QPixmap::fromImage(image_));
}

void Painter::mousePressEvent(QMouseEvent *event) {
  button_ = event->button();
  if (button_ == Qt::LeftButton) {
    start_ = event->pos();
  } else if (button_ == Qt::RightButton) {
    ClearDrawingArea();
  }
}

void Painter::mouseMoveEvent(QMouseEvent *event) {
  if (button_ == Qt::LeftButton) {
    Draw(event->pos());
  }
}

void Painter::Draw(const QPoint &end) {
  QPainter painter(&image_);

  QConicalGradient gradient;
  gradient.setCenter(rect().topRight() / 2);
  gradient.setAngle(47.28);
  gradient.setColorAt(0.25, QColor(68, 235, 153));
  gradient.setColorAt(1.0, QColor(134, 27, 227));
  QPen pen = QPen(gradient, 0);

  pen.setStyle(Qt::DotLine);      // сплошная линия
  pen.setCapStyle(Qt::RoundCap);  // округлое перо

  pen.setWidth(28 * 2.5);  // толщина
  painter.setPen(pen);

  painter.drawLine(start_, end);

  start_ = end;
  update();
}

void Painter::ClearDrawingArea() {
  image_.fill(Qt::white);
  update();
}

void Painter::ConvertPictureToArray() {
  std::vector<double> pic_data_;
  QImage resizedImg = image_.scaled(kSIZE, kSIZE, Qt::KeepAspectRatio,
                                    Qt::SmoothTransformation);
  resizedImg.invertPixels(QImage::InvertRgba);

  for (int x = 0; x < kSIZE; x++) {
    for (int y = 0; y < kSIZE; y++) {
      QColor color = resizedImg.pixelColor(x, y);
      float r, g, b;
      r = color.red();
      g = color.green();
      b = color.blue();
      int gray = (r + g + b) / 3;
      pic_data_.push_back(gray);
    }
  }

  emit signalSendDataToRecognize(pic_data_);
}
