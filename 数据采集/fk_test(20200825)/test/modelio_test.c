#include <stdio.h>
#include <stdlib.h>

int model_init(void)
{
    printf("Module was initialized!\n");
    return 0;
}

int model_rst(void)
{
    printf("Module was reset!\n");
    return 0;
}

int model_close(void)
{
    printf("Module was closed!\n");
    return 0;
}

int model_reg_read(int modeladdr,int regaddr,int regnum,int *regdata)
{
    int i = 0;

    for(i=0;i<regnum;i++)
    {
        regdata[i] = rand() % 200;
        regdata[i] -= 100;
    }

    return i;
}