import numpy as np

class State:
    def __init__(self, data, gamma=1.4):
        if isinstance(data, np.ndarray):
            self.data = data
        else:
            self.data = np.array(data)
            
        self.x = np.linspace(0, 1, len(self.data))
        self.dx = self.x[1]
        self.dt = 2.5e-3
        self.gamma = gamma

    def rho(self): return self.data.T[0]
    def m(self): return self.data.T[1]
    def E(self): return self.data.T[2]

    def P(self): return (self.E() - 0.5 * self.m()**2 / self.rho()) * (self.gamma - 1)
    def u(self): return self.m() / self.rho()
    def e(self): return self.E() / self.rho() - 0.5 * (self.m() / self.rho()) ** 2

    def c(self): return np.sqrt(np.abs(self.gamma * self.P() / self.rho()))
    def sigma(self): return np.max(np.abs(self.u()) + self.c()) * self.dt / self.dx

    def flux(self):
        return np.array([
            self.m(),
            self.m() ** 2 / self.rho() + self.P(),
            (self.m() / self.rho()) * (self.E() + self.P())
        ]).T