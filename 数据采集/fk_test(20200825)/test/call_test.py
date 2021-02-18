from ctypes import cdll

cur = cdll.LoadLibrary('./mod_test.so')

cur.model_init()

cur.model_close()
