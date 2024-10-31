import numpy as np
import numpy.linalg as LA
from schemes import update
from state import State
from schemes import shift

# 1D, fully discrete, 2nd order
def FD2_update(state, params = None):
    U_n, dx, dt = state.data.copy(), state.dx, state.dt
    F_n = state.flux().copy()
    l, N = dt/dx, len(U_n)

    minmod = lambda a, b : 0.5 * (np.sign(a) + np.sign(b)) * np.minimum(np.abs(a), np.abs(b))
    Dx = lambda U : minmod((U - shift(U, -1))/dx , (shift(U, 1) - U)/dx)

    U_x = Dx(U_n)

    # local speeds
    state_plus = State(shift(U_n, 1) - 0.5 * dx * shift(U_x, 1))
    state_minus = State(U_n + 0.5 * dx * U_x)

        # jacobians of +/- states
    FU_plus, FU_minus = state_plus.f_x().copy(), state_minus.f_x().copy()
    rho = lambda A : np.array([[np.max(np.abs(LA.eigvals(a))) for a in A]])

    a = np.maximum(rho(FU_minus), rho(FU_plus)).T

    # midvalues
    U_n_jhalf_l = U_n + dx * U_x * (0.5 - l * a)
    U_n_jhalf_r = shift(U_n, 1) - dx * shift(U_x, 1) * (0.5 - l * a)

        # flux derivatives, componentwise
    Fx_l_forward = (State(shift(U_n_jhalf_l, 1)).flux() - State(U_n_jhalf_l).flux()) / dx
    Fx_r_forward = (State(shift(U_n_jhalf_r, 1)).flux() - State(U_n_jhalf_r).flux()) / dx

    Fx_l_backward = (State(U_n_jhalf_l).flux() - State(shift(U_n_jhalf_l, -1)).flux()) / dx
    Fx_r_backward = (State(U_n_jhalf_r).flux() - State(shift(U_n_jhalf_r, -1)).flux()) / dx

    Fx_l = minmod(Fx_l_forward, Fx_l_backward)
    Fx_r = minmod(Fx_r_forward, Fx_r_backward)

        # flux derivatives, exact
    # Fx_l = np.array([np.matmul(State(U_n_jhalf_l).f_x()[i], Dx(U_n_jhalf_l)[i]) for i in range(N)])
    # Fx_r = np.array([np.matmul(State(U_n_jhalf_r).f_x()[i], Dx(U_n_jhalf_r)[i]) for i in range(N)])

    U_nhalf_jhalf_l = U_n_jhalf_l - 0.5 * dt * Fx_l
    U_nhalf_jhalf_r = U_n_jhalf_r - 0.5 * dt * Fx_r

    F_nhalf_jhalf_l = State(U_nhalf_jhalf_l).flux().copy()
    F_nhalf_jhalf_r = State(U_nhalf_jhalf_r).flux().copy()
    F_nmhalf_jhalf_r = State(shift(U_nhalf_jhalf_r, -1)).flux().copy()

    # reconstructed slopes
    w_jhalf = (0.5 * (U_n + shift(U_n, 1)) + 0.25 * (dx - a * dt) * (U_x - shift(U_x, 1))
               - 0.5 * (F_nhalf_jhalf_r - F_nhalf_jhalf_l) / a)
    w_j = (U_n + 0.5 * dt * (shift(a, -1) - a) * U_x
           - l * (F_nhalf_jhalf_l - F_nmhalf_jhalf_r) / (1 - l * (shift(a,-1) + a)))
    
    mm1 = (shift(w_j, 1) - w_jhalf) / (1 + l * (a - shift(a, 1)))
    mm2 = (w_jhalf - w_j) / (1 + l * (a - shift(a, -1)))
    
    U_x_jhalf = (2/dx) * minmod(mm1, mm2)

    #  fully discrete 2nd order
    term1 = l * shift(a, -1) * shift(w_jhalf, -1) + l * a * w_jhalf
    term2 = (1 - l * (shift(a, -1) + a)) * w_j
    term3 = 0.5 * dx * (((l * shift(a, -1))**2 * shift(U_x_jhalf, -1) - (l * a) ** 2 * U_x_jhalf))

    state.data = term1 + term2 + term3

        # rusanov update
    # state.data = U_n - l * 0.5 * (shift(F_n, 1) - shift(F_n, -1)) + 0.5 * l * (a * (shift(U_n, 1)  - U_n) - shift(a, -1) * (U_n - shift(U_n, -1)))


def SD2_update(state, params = None):
    pass
    # Un, dt, dx = state.data.copy(), state.dt, state.dx
    # minmod = lambda a, b : 0.5 * (np.sign(a) + np.sign(b)) * np.minimum(np.abs(a), np.abs(b))
    # Dx = lambda U : minmod((U - shift(U, -1))/dx , (shift(U, 1) - U)/dx)

    # if hasattr(state, 'Q'):
    #     Q = state.Q()
    # else: Q = lambda: 0
    
    # Ux = Dx(Un)

    # state_plus = State(shift(Un, 1) - 0.5 * dx * shift(Ux, 1))
    # state_minus = State(Un + 0.5 * dx * Ux)

    # Uplus, Uminus = state_plus.data, state_minus.data
    # Fplus, Fminus = state_plus.flux(), state_minus.flux()
    # Fxplus, Fxminus = state_plus.f_x(), state_minus.f_x()

    # a = np.maximum(np.abs(Fxplus), np.abs(Fxminus))

    # H = 0.5 * (Fplus + Fminus) - 0.5 * a * (Uplus - Uminus)

    # Ux_apx = (shift(Un, 1) - Un) / dx

    # P = 0.5 * (Q(Un, Ux_apx) + Q(shift(Un, 1), Ux_apx))

    # dUdx = (shift(H, -1) - H + P - shift(P, 1)) / dx

    # state.data = Un + dt * dUdx

