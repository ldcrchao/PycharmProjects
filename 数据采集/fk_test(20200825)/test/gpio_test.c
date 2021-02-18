#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <string.h>
#include <sys/ioctl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>

#define FATAL do { fprintf(stderr, "Error at line %d, file %s (%d) [%s]\n", \
  __LINE__, __FILE__, errno, strerror(errno)); exit(1); } while(0)

#define MAP_SIZE 4096UL
#define MAP_MASK (MAP_SIZE - 1)

volatile int LED = 0;
volatile int CMD = 0;
volatile int RST = 0;

int fd_led = 0;
int fd_cmd = 0;
int fd_rst = 0;

int fd_mem = 0;

void set_off_on(int fd, volatile int* gpio);

int main(void)
{
    void *map_base, *virt_addr;
    unsigned long read_result, write_val, cmd_val;
    unsigned long only_read, only_write;
    off_t target = 0x08000000;

    fd_led = open("/dev/leds_ctl", 0);

    if(fd_led < 0)
    {
        perror("open device led failed!");
        return -1;
    }

    fd_cmd= open("/dev/model_cmd_pin", 0);

    if(fd_cmd < 0)
    {
        perror("open device cmd pin failed!");
        return -1;
    }

    fd_rst= open("/dev/model_rst_pin", 0);

    if(fd_rst < 0)
    {
        perror("open device reset pin failed!");
        return -1;
    }

    if((fd_mem = open("/dev/mem", O_RDWR | O_SYNC)) == -1) FATAL;
    printf("/dev/mem opened.\n");
    fflush(stdout);

    /* Map one page */
    map_base = mmap(0, MAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd_mem, target & ~MAP_MASK);
    if(map_base == (void *) -1) FATAL;
    printf("Memory mapped at address %p.\n", map_base);
    fflush(stdout);

    virt_addr = map_base + (target & MAP_MASK);

    write_val = 26;
    cmd_val = 29;
    only_write = 0xaaaa;

    printf("Please input a char(l, c, r, t, n, f, g, x):\n");
    printf("--l(led) c(cmd) r(reset) t(test)\n");

    printf("--f(only read) g(only write)\n");
    printf("--input x exit program\n");

    while(1)
    {
        switch(getchar())
        {
            case 'l':
            {
                printf("Key \'led\' is down.\n");
                set_off_on(fd_led, &LED);
            }
            break;

            case 'c':
            {
                printf("Key \'cmd\' is down.\n");
                set_off_on(fd_cmd, &CMD);
            }
            break;

            case 'r':
            {
                printf("Key \'reset\' is down.\n");
                set_off_on(fd_rst, &RST);
            }
            break;

            case 's':
            {
                printf("Key \'reset test\' is down.\n");
                ioctl(fd_rst, 1, 0);
                usleep(100*1000);
                ioctl(fd_rst, 0, 0);
                usleep(100*1000);
                ioctl(fd_rst, 1, 0);
            }
            break;

            case 't':
            {
                printf("Key \'test\' is down.\n");

                ioctl(fd_cmd, 1, 0);

                usleep(10);

                *((unsigned short *) virt_addr) = write_val;

                usleep(10);

                ioctl(fd_cmd, 0, 0);

                usleep(10);

                read_result = *((unsigned short *) virt_addr);

                printf("Written 0x%X; readback 0x%X\n", write_val, read_result);

                fflush(stdout);
            }
            break;
            
            case '1':
            {
                printf("Key \'test\' is down.\n");

                ioctl(fd_cmd, 1, 0);

                usleep(100);

                *((unsigned short *) virt_addr) = write_val;

                usleep(100);

                ioctl(fd_cmd, 0, 0);

                usleep(100);

                read_result = *((unsigned short *) virt_addr);

                printf("Written 0x%X; readback 0x%X\n", write_val, read_result);

                fflush(stdout);
            }
            break;

            case '2':
            {
                printf("Key \'test\' is down.\n");

                ioctl(fd_cmd, 1, 0);

                usleep(1000);

                *((unsigned short *) virt_addr) = write_val;

                usleep(1000);

                ioctl(fd_cmd, 0, 0);

                usleep(1000);

                read_result = *((unsigned short *) virt_addr);

                printf("Written 0x%X; readback 0x%X\n", write_val, read_result);

                fflush(stdout);
            }
            break;

            case 'n':
            {
                printf("Key \'name\' is down.\n");

                ioctl(fd_cmd, 1, 0);

                usleep(1);

                *((unsigned short *) virt_addr) = cmd_val;

                usleep(1);

                ioctl(fd_cmd, 0, 0);

                usleep(1);

                read_result = *((unsigned short *) virt_addr);

                printf("Written 0x%X; readback 0x%X\n", cmd_val, read_result);

                fflush(stdout);
            }
            break;

            case 'f':
            {
                printf("Key \'only read\' is down.\n");

                read_result = *((unsigned short *) virt_addr);

                printf("Read Value: 0x%X\n", read_result);

                fflush(stdout);
            }
            break;

            case 'g':
            {
                printf("Key \'only write\' is down.\n");

                *((unsigned short *) virt_addr) = only_write;

                printf("Write data 0x%X\n", only_write);

                fflush(stdout);
            }
            break;

            case 'x':
            {
                return 0;
            }
            break;

            case '\r':
            case '\n':
            break;

            default:
            {
                printf("Unknown command.\n");
            }
            break;
        }
    }

    close(fd_led);
    close(fd_cmd);
    close(fd_rst);

    if(munmap(map_base, MAP_SIZE) == -1) FATAL;
    close(fd_mem);

    return 0;
}

void set_off_on(int fd, volatile int *gpio)
{
    *gpio = ~(*gpio);

    if(*gpio == 0)
    {
        ioctl(fd, 1, 0);
    }
    else
    {
        ioctl(fd, 0, 0);
    }
}
