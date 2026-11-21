import sys, torch
torch.set_num_threads(4)
from doppel_lab import allie_ttt
allie_ttt.CHUNK = 2
allie_ttt.run(1500, "cpu", 2, 1, 1e-3, 0.01, 512, 4, "")
