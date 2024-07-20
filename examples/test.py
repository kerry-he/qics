from cones import *
import numpy as np
from utils import quantum as quant
import time

n = 10
K = opperspecepi.Cone(100, "log")

pnt0 = K.zeros()
pnt1 = K.zeros()
K.get_init_point(pnt0)
K.get_feas()
K.grad_ip(pnt1)
print()