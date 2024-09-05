#include <QApplication>
#include <QDir>
#include <QIcon>
#include <QStandardPaths>
#include <QTemporaryDir>

#include "controller/controller.h"
#include "view/mainwindow.h"

int main(int argc, char *argv[]) {
  QApplication a(argc, argv);
  QCoreApplication::setAttribute(Qt::AA_DontUseNativeMenuBar);

  Q_INIT_RESOURCE(resource);
  a.setWindowIcon(QIcon(":/img/icons/app_icon.png"));

  cpp_mlp::Controller ctrl;
  cpp_mlp::MainWindow w(&ctrl);
  w.show();

  return a.exec();
}
