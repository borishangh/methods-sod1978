import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

def exact(t, N, dx, gamma=1.4):
    p1, u1, rho1 = 1.0, 0.0, 1.0
    p5, u5, rho5 = 0.1, 0.0, 0.125
    x0 = 0.5
    alpha = (gamma + 1) / (gamma - 1)
    cR = np.sqrt(gamma * p5 / rho5)
    cL = np.sqrt(gamma * p1 / rho1)

    def equation(p45):
        return (
            1 + float((u1 - u5) / cL) * (gamma - 1) / 2
            - ((gamma - 1) * float(cR / cL) * (p45 - 1))
            / np.sqrt(2 * gamma * (gamma - 1 + (gamma + 1) * p45))
        ) ** (2 * gamma / (gamma - 1)) / p45 - float(p5 / p1)

    p45 = fsolve(equation, np.array(3.0))[0]  # p45 = p4 / p5
    p4 = p45 * p5  # p4 = p3
    rho4 = rho5 * (1 + alpha * p45) / (alpha + p45)
    rho3 = rho1 * (p4 / p1) ** (1 / gamma)
    u3 = u1 + (2 * cL / (gamma - 1)) * (
        1 - (p4 / p1) ** ((gamma - 1) / (2 * gamma))
    )  # u3 = u4
    c3 = np.sqrt(gamma * p4 / rho3)

    pos1 = x0 + (u1 - cL) * t  # Head of rarefaction wave
    pos2 = x0 + (u3 - c3) * t  # Tail of rarefaction wave
    conpos = x0 + u3 * t  # Contact discontinuity
    spos = (
        x0
        + t * cR * np.sqrt((gamma - 1) / (2 * gamma) + (gamma + 1) / (2 * gamma) * p45)
        + t * u5
    )  # #Shock wave
    exact = np.zeros((N, 4))
    for i in range(0, N):
        if i * dx <= pos1:  # region 1
            exact[i, 0] = rho1
            exact[i, 1] = u1
            exact[i, 2] = p1
        elif i * dx <= pos2:  # region 2
            exact[i, 0] = rho1 * (1 - ((i * dx - x0) / t + cL) / (cL * alpha)) ** (
                2 / (gamma - 1)
            )
            exact[i, 1] = u1 + (2 / (gamma + 1)) * ((i * dx - x0) / t + cL)
            exact[i, 2] = p1 * (1 - ((i * dx - x0) / t + cL) / (cL * alpha)) ** (
                2 * gamma / (gamma - 1)
            )
        elif i * dx <= conpos:  # #region 3
            exact[i, 0] = rho3
            exact[i, 1] = u3
            exact[i, 2] = p4
        elif i * dx <= spos:  # #region 4
            exact[i, 0] = rho4
            exact[i, 1] = u3
            exact[i, 2] = p4
        else:  # region 5
            exact[i, 0] = rho5
            exact[i, 1] = u5
            exact[i, 2] = p5
    exact[:, 3] = (
        1 / (gamma - 1) * exact[:, 2] / exact[:, 0]
    )  # internal energy per unit mass
    return exact

def plot(computed, t, heading, limits = False):
    fig, axs = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle(heading, fontsize=12)
    y_labels = ['DENSITY', 'VELOCITY', 'PRESSURE', 'ENERGY']

    N = len(computed.data)
    dx = 1/(N-1)
    y_exact = exact(t, N, dx)

    x = np.linspace(0,1, N)
    y_comp = [computed.rho(), computed.u(), computed.P(), computed.e()]
    ylim_values = [(0, 1.1), (0, 1.1), (-0.1, 1.1), (3, 1.5)]
    
    for i, ax in enumerate(axs.flat):
        ax.plot(x, y_exact.T[i], 'r')
        ax.plot(x, y_comp[i], 'kx')
        ax.set_ylabel(y_labels[i])
        if limits:
            ax.set_ylim(ylim_values[i])

    plt.tight_layout(rect=[0, 0, 1, 0.99]) 
    plt.show()

def shift(a, n):
    N = len(a)
    if n > 0:
        nth = a[-1]
        shifted = np.array([a[i + n] if i < N - n - 1 else nth for i in range(N)])
    else:
        nth = a[0]
        shifted = np.array([a[i + n-1] if i > - n else nth for i in range(N, 0, -1)])[::-1]
    return shifted

def evolve(state, t, update, params):
    steps = int(t // state.dt)
    for i in range(steps):
        update(state, params)
    # return state