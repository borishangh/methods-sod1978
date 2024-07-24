from state import State
import numpy as np
from helper import shift

def AV(state, nu=1):
    U, l = state.data.copy(), state.dt / state.dx
    u = np.array(state.m() / state.rho()).reshape(len(U), 1)
    
    new_state = State(U + nu*l*(
        np.abs(shift(u, 1) - u) * (shift(U, 1) - U)
        - np.abs(u - shift(u, 1)) * (U - shift(U, -1))
    ))

    state.data = new_state.data