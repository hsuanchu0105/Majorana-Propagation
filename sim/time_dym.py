from src.MajProp import * 
from Setting.Setting_td import *
from src.Gate import *
from src.Err_anlys import *
from src.RMP import *
from src.TE import *
import time 
from datetime import date
from datetime import datetime 
from pathlib import Path



t_st = 0
dt = 0.1
ts = 10
t_end = t_st + dt * ts

mp = np.zeros(ts)
rmp = np.zeros(ts)
ana = np.zeros(ts)

