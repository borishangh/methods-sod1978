import numpy as np
from functools import wraps
from helper import shift
from state import State
from artificial_viscocity import AV
import numpy.linalg as LA
from artificial_compression import ACM, theta_hybrid

def update(func):
    @wraps(func)
    def wrapper(state, params=None):
        if params is None:
            params = {}

        params.setdefault("ACM", False)
        params.setdefault("AV", False)

        if params["ACM"]:
            ACM(state, l=1 if params["ACM_lambda"] is None else params["ACM_lambda"], 
                switch=False if params["Hybrid_switch"] is None else params["Hybrid_switch"])
        if params["AV"]:
            AV(state, nu=1 if params["AV_nu"] is None else params["AV_nu"])
        
        return func(state, params)
    return wrapper

@update
def godunov_update(state, params):
    U_n, F_n, l = state.data.copy(), state.flux().copy(), state.dt / state.dx

    temp_state = State(0.5 * (shift(U_n, 1) + U_n) - l * (shift(F_n, 1) - F_n))
    Fbar = temp_state.flux().copy()

    state.data = U_n - l * (Fbar - shift(Fbar, -1))
    
@update
def laxwendroff_update(state, params):
    U_n, F_n, l = state.data.copy(), state.flux().copy(), state.dt / state.dx

    temp_state = State(0.5 * (shift(U_n, 1) + U_n) - 0.5 * l * (shift(F_n, 1) - F_n))
    Fbar = temp_state.flux().copy()

    state.data = U_n - l * (Fbar - shift(Fbar, -1))

@update
def maccormack_update(state, params):
    U_n, F_n, l = state.data.copy(), state.flux().copy(), state.dt / state.dx

    temp_state = State(U_n - l * (shift(F_n, 1) - F_n))

    Ubar = temp_state.data.copy()
    Fbar = temp_state.flux().copy()

    state.data = 0.5 * (U_n + Ubar) - 0.5 * l * (Fbar - shift(Fbar, -1))

@update
def rusanov_update(state, params):
    U, F, l = state.data.copy(), state.flux().copy(), state.dt / state.dx
    dx = state.dx
    omega = 1 or params["Rusanov_omega"]

    alpha = np.array(omega * l * (state.u() + state.c())).reshape(len(U), 1)

    state.data = (
        U - 0.5 * l * (shift(F, 1) - shift(F, -1)) 
        + 0.25 * (shift(alpha, 1) + alpha) * (shift(U, 1) - U) 
        - 0.25 * (alpha + shift(alpha, -1))*(U - shift(U, -1))
    )

@update
def hybrid_update(state, params):
    U_n, F_n, l = state.data.copy(), state.flux().copy(), state.dt / state.dx
    rho = state.rho().copy()

    theta = theta_hybrid(rho)

    temp_state = State(U_n - l * (shift(F_n, 1) - F_n))

    Ubar = temp_state.data.copy()
    Fbar = temp_state.flux().copy()

    state.data = (
        0.5 * (U_n + Ubar) - 0.5 * l * (Fbar - shift(Fbar, -1)) 
        + 0.125 * (theta * (shift(U_n, 1) - U_n) 
        - shift(theta, -1) * (U_n - shift(U_n, -1))))

@update
def hyman_update(state, params):
    U_n, F_n= state.data.copy(), state.flux().copy()
    u, c, dt, dx = state.u().copy(), state.c().copy(), state.dt, state.dx
    delta = 0.8 or params["Hyman_delta"]
    D = lambda F : (1/12) * (-shift(F, 2) + 8*shift(F, 1) - 8*shift(F, -1) + shift(F, -2)) / dx

    alpha = (u + c).reshape((len(u), 1))
    phi = 0.25 * (shift(alpha, 1) + alpha) * (shift(U_n, 1) - U_n) / dx
    if params["Hyman_switch"]: 
        beta = np.where(shift(alpha, 1) > alpha + dx/3, 1/3, 1)
        phi *= beta
    
    P = D(F_n ) - delta * (phi - shift(phi, -1))

    temp_state = State(U_n - dt * P)
    Fhalf = temp_state.flux().copy()

    state.data = U_n - 0.5 * dt * (D(Fhalf) + P)

def antidiffusion_update(state, params):
    U = state.data.copy()
    eta = 0.125 or params["diffusion_coefficient"]
    
    laxwendroff_update(state)
    Utilde = state.data.copy()
    Deltahat = eta * (shift(Utilde, 1) - Utilde)

    temp_state = State(Utilde + eta * (shift(U, 1) - 2 * U + shift(U, -1)))
    Uhat = temp_state.data.copy()
    Delta = shift(Uhat, 1) - Uhat

    min1 = np.sign(Deltahat) * shift(Delta, -1)
    min2 = np.abs(Deltahat)
    min3 = np.sign(Deltahat) * Delta
    min_term = np.minimum(np.minimum(min1, min2), min3)

    f = np.sign(Deltahat)*np.maximum(np.zeros(Delta.shape), min_term)

    state.data = Uhat - (f - shift(f, -1))