from state import State
from helper import evolve, plot, exact
from schemes import *
from kurganov_tadmor import FD2_update

import matplotlib.pyplot as plt

N, t = 100, 0.16

    # sod
# stateL = [1.0, 0.0, 2.5]
# stateR = [0.125, 0.0, 0.25]

    # lax
stateL = [0.445, 0.311, 8.928]
stateR = [0.5, 0, 1.4275]

init = [stateL] * (N // 2) + [stateR] * (N // 2)

tube = State(init)
l = tube.dt / tube.dx

params = {
    "ACM" : False,
    "ACM_lambda" : 1, 
    "Hybrid_switch": True,
    "AV": False,
    "AV_nu": 1,
    "Rusanov_omega" : 1,
    "Hyman_delta" : 0.8,
    "Hyman_switch" : False,
}

# godunov_update, laxwendroff_update, maccormack_update,
# rusanov_update, hybrid_update, hyman_update, antidiffusion_update

evolve(tube, t, hyman_update, params)

plot(tube, t, stateL, stateR)