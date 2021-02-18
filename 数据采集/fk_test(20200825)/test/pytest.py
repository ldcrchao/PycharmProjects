import os
import ctypes
from ctypes import cdll

cur = ctypes.cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), "./modeio.so"))

cur.model_init()

cur.model_close()
