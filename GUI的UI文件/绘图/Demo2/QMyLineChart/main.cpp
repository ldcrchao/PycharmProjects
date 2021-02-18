#include "QMyLineChart.h"

#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    QMyLineChart w;
    w.show();
    return a.exec();
}
