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

void *map_base, *virt_addr;

int model_init(void)
{
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
	return 1;
}

int model_close(void)
{
    	close(fd_led);
    	close(fd_cmd);
    	close(fd_rst);

    	if(munmap(map_base, MAP_SIZE) == -1) FATAL;
    	close(fd_mem);

    	return 1;
}

int model_rst(void)
{
    	ioctl(fd_rst, 1, 0);
    	usleep(100);
    	ioctl(fd_rst, 0, 0);
    	usleep(100);
    	ioctl(fd_rst, 1, 0);

    	return 1;
}

int model_reg_read(unsigned short modeladdr,unsigned short regaddr,unsigned short regnum,unsigned short *regdata)
{
        unsigned short i,cmd_val;

        for(i=0;i<regnum;i++)
	{
        	ioctl(fd_cmd, 1, 0);
                cmd_val = (modeladdr<<5)|(regaddr+i);
        	*((unsigned short *) virt_addr) = cmd_val;
        	ioctl(fd_cmd, 0, 0);
        	regdata[i] = *((unsigned short *) virt_addr);
                printf("model 0x%X reg read is 0x%X\n",cmd_val,regdata[i]);
		fflush(stdout);
 	}
        return 1;
}

int model_reg_write(unsigned short modeladdr,unsigned short regaddr,unsigned short regnum,unsigned short *regdata)
{
        unsigned short i,cmd_val;

	for(i=0;i<regnum;i++)
	{
		ioctl(fd_cmd, 1, 0);
                cmd_val = (modeladdr<<5)|(regaddr+i);
        	*((unsigned short *) virt_addr) = 0x8000|cmd_val;
        	ioctl(fd_cmd, 0, 0);
        	*((unsigned short *) virt_addr) = regdata[i];
                printf("model 0x%X reg write is 0x%X\n",cmd_val, regdata[i]);
        	fflush(stdout);
	}
        return 1;
}

int model_fifo_read(unsigned short modeladdr,unsigned short regaddr,unsigned short datanum,unsigned char *regdata)
{
	int i;
	unsigned short cmd_val,data_temp;

        ioctl(fd_cmd, 1, 0);
	cmd_val = (modeladdr<<5)|regaddr;
        *((unsigned short *) virt_addr) = cmd_val;
        ioctl(fd_cmd, 0, 0);
        for(i=0;i<datanum/2;i++)
	{
        	data_temp = *((unsigned short *) virt_addr);
		regdata[i*2]   = data_temp>>8;
		regdata[i*2+1] = data_temp&0x00ff;
		printf("model 0x%X reg read is 0x%X\n",cmd_val,data_temp);
	}
	fflush(stdout);
        return 1;
}

int model_fifo_read_short(unsigned short modeladdr,unsigned short regaddr,unsigned short datanum,unsigned short *regdata)
{
	int i;
	unsigned short cmd_val,data_temp;
        unsigned int fifo_data_num,fifo_data_num_h,fifo_data_num_l;

	while(1)
        {
                ioctl(fd_cmd, 1, 0);
                cmd_val = (modeladdr<<5)|17;
                *((unsigned short *) virt_addr) = cmd_val;
                ioctl(fd_cmd, 0, 0);
                fifo_data_num_h = *((unsigned short *) virt_addr);

                ioctl(fd_cmd, 1, 0);
                cmd_val = (modeladdr<<5)|18;
                *((unsigned short *) virt_addr) = cmd_val;
                ioctl(fd_cmd, 0, 0);
                fifo_data_num_l = *((unsigned short *) virt_addr);

                fifo_data_num = (fifo_data_num_h<<16)|fifo_data_num_l;
                //printf("model fifo_data_num read is 0x%X\n",fifo_data_num);
                if(fifo_data_num>datanum)
                {
                        break;
                }
        }

        ioctl(fd_cmd, 1, 0);
	cmd_val = (modeladdr<<5)|regaddr;
        *((unsigned short *) virt_addr) = cmd_val;
        ioctl(fd_cmd, 0, 0);
        for(i=0;i<datanum;i++)
	{
        	data_temp = *((unsigned short *) virt_addr);
		//regdata[i*2]   = data_temp>>8;
		//regdata[i*2+1] = data_temp&0x00ff;
		regdata[i] = data_temp;
                printf("model 0x%X reg read is 0x%X\n",cmd_val,data_temp);
	}
	fflush(stdout);
        return 1;
}

int model_fifo_read_int(unsigned short modeladdr,unsigned short regaddr,unsigned short datanum,unsigned int *regdata)
{
	int i;
	unsigned short cmd_val,data_temp;
        unsigned int adtemp,temp1,temp2,temp3,temp4;
	unsigned int fifo_data_num,fifo_data_num_h,fifo_data_num_l;

	while(1)
	{
		ioctl(fd_cmd, 1, 0);
                cmd_val = (modeladdr<<5)|17;
        	*((unsigned short *) virt_addr) = cmd_val;
        	ioctl(fd_cmd, 0, 0);
        	fifo_data_num_h = *((unsigned short *) virt_addr);

		ioctl(fd_cmd, 1, 0);
                cmd_val = (modeladdr<<5)|18;
        	*((unsigned short *) virt_addr) = cmd_val;
        	ioctl(fd_cmd, 0, 0);
        	fifo_data_num_l = *((unsigned short *) virt_addr);

		fifo_data_num = (fifo_data_num_h<<16)|fifo_data_num_l;
                //printf("model fifo_data_num read is 0x%X\n",fifo_data_num);
		if(fifo_data_num>(datanum*2))
		{
			break;
		}
	}

        ioctl(fd_cmd, 1, 0);
	cmd_val = (modeladdr<<5)|regaddr;
        *((unsigned short *) virt_addr) = cmd_val;
        ioctl(fd_cmd, 0, 0);
        for(i=0;i<datanum;i++)
	{
        	data_temp = *((unsigned short *) virt_addr);
		temp1  = data_temp>>8;
		temp2  = data_temp&0x00ff;
                data_temp = *((unsigned short *) virt_addr);
		temp3  = data_temp>>8;
		temp4  = data_temp&0x00ff;
                adtemp = (temp1<<24)|(temp2<<16)|(temp3<<8)|temp4;
                regdata[i] = adtemp;
		//printf("model 0x%X reg read is 0x%X,0x%X,0x%X,0x%X,0x%X\n",cmd_val,adtemp,temp1,temp2,temp3,temp4);
	}
	fflush(stdout);
        return 1;
}

int model_fifo_write(unsigned short modeladdr,unsigned short regaddr,unsigned short datanum,unsigned char *regdata)
{
        int i;
	unsigned short data_temp;

	ioctl(fd_cmd, 1, 0);
        *((unsigned short *) virt_addr) = 0x8000|(modeladdr<<5)|regaddr;
        ioctl(fd_cmd, 0, 0);
	for(i=0;i<datanum/2;i++)
	{
		data_temp = regdata[i*2];
		data_temp = data_temp<<8;
		data_temp = data_temp|regdata[i*2+1];
        	*((unsigned short *) virt_addr) = data_temp;
	}
        fflush(stdout);
        return 1;
}

