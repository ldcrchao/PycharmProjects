#include "QMyTreeWidget.h"

#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    QMyTreeWidget w;
    w.show();
    return a.exec();
}
