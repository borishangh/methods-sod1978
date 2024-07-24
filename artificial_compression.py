import numpy as np
from helper import shift
from state import State


def ACM(state, l=1, switch=False):
    U = state.data.copy()
    delta = shift(U, 1) - U
    S = np.sign(delta)

    alpha = np.zeros((len(U)))

    comp = np.array([
            np.min(np.array([np.abs(delta)[..., i], 
                             (shift(delta, -1) * np.sign(delta))[..., i]]).T,
                axis=1,) for i in range(3)]).T

    div = np.abs(delta) + np.abs(shift(delta, -1))
    div[div == 0] = 1  # to not div by zero
    
    alpha = np.max(np.array([alpha, np.min(comp / div, axis=1)]).T, axis=1)

    Delta = shift(U, 1) - shift(U, -1)
    g = (alpha).reshape((len(alpha), 1)) * Delta
    G = g + shift(g, 1) - np.abs(shift(g, 1) - g) * S

    if switch:
        theta = theta_hybrid(state.rho().copy())
        new_state = State(U - 0.5 * l * (G * theta - shift(G, -1) * shift(theta, -1)))
    else:
        new_state = State(U - 0.5 * l * (G - shift(G, -1)))

    state.data = new_state.data
