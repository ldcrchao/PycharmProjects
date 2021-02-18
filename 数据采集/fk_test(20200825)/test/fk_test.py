import time
from ctypes import * 

cur = cdll.LoadLibrary('./modelio.so')

cur.model_init()

cur.model_rst()

#buff = range(100)
temp = c_ushort(0)
p_temp = pointer(temp)
cur.model_reg_read(0,21,1,p_temp)
print(p_temp[0])
p_temp[0] = 0xAAAA
cur.model_reg_write(0,21,1,p_temp)
p_temp[0] = 0x0000
cur.model_reg_read(0,21,1,p_temp)
print(p_temp[0])

p_temp[0] = 0  
cur.model_reg_write(0,10,1,p_temp)
time.sleep(0.1)

p_temp[0] = 0xFFFF
cur.model_reg_write(0,1,1,p_temp)
p_temp[0] = 256 #ad fre is 1000 
cur.model_reg_write(0,3,1,p_temp)
p_temp[0] = 0x8900 
cur.model_reg_write(0,9,1,p_temp)
p_temp[0] = 0 
cur.model_reg_write(0,11,1,p_temp)
time.sleep(0.01) #delay 0.01s
p_temp[0] = 1 
cur.model_reg_write(0,11,1,p_temp)
time.sleep(0.01)
p_temp[0] = 0  
cur.model_reg_write(0,11,1,p_temp)
time.sleep(0.01)
p_temp[0] = 1  
cur.model_reg_write(0,11,1,p_temp)

time.sleep(0.1)
p_temp[0] = 1  
cur.model_reg_write(0,10,1,p_temp)

time.sleep(1) #delay 1s

buff = (c_ubyte*1000)()
for i in xrange(0,1000):
	buff[i] = 0 
cur.model_fifo_read(0,19,1000,buff)
for i in xrange(0,64):
	print(buff[i])

cur.model_close()
