from state import State
from helper import evolve, plot
from schemes import *


N, t = 100, 0.2
sod_init = [[1.0, 0.0, 1.0 / 0.4]] * (N // 2) + [[0.125, 0.0, 0.1 / 0.4]] * (N // 2)

sod_state = State(sod_init)
l = sod_state.dt / sod_state.dx

params = {
    "ACM" : True,
    "ACM_lambda" : 1, 
    "Hybrid_switch": False,
    "AV": False,
    "AV_nu": 1,
    "Rusanov_omega" : 1,
    "Hyman_delta" : 0.8,
}

# godunov_update, laxwendroff_update, maccormack_update,
# rusanov_update, hybrid_update, hyman_update

evolve(sod_state, t, rusanov_update, params)
plot(sod_state, t, r"Rusanov Scheme with ACM $\hat{\lambda} = 1$ and Hybrid Switch")