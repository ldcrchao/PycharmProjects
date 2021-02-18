#include "QComboBox.h"

#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    QComboBox w;
    w.show();
    return a.exec();
}
