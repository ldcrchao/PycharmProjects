#include "stdio.h"

int model_init(void)
{
    printf("Module was initialized!\n");
    return 0;
}

int model_close(void)
{
    printf("Module was closed!\n");
    return 0;
}