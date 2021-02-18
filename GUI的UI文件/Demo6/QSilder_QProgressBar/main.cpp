#include "QSilder_QProgressBar.h"

#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    QSilder_QProgressBar w;
    w.show();
    return a.exec();
}
