# -*- coding: utf-8 -*- 
# @Time : 2022/11/17 14:55 
# @Author : Yinan 
# @File : pH_model.py

from scipy.integrate import solve_bvp, solve_ivp, odeint
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import input_generate

class ph_model():
    def __init__(self, simulation_time, sample_time):
        """the nominal values of the model parameters"""
        self.z = 11.5
        self.C_v4 = 4.59
        self.n = 0.607
        self.pK1 = 6.35
        self.pK2 = 10.25
        self.h1 = 14
        self.pH = 7
        self.W_a1 = 3e-3
        self.W_b1 = 0
        self.W_a2 = -0.03
        self.W_b2 = 0.03
        self.W_a3 = 3.05e-3
        self.W_b3 = 5e-5
        self.W_a4 = -4.32e-4
        self.W_b4 = 5.28e-4
        self.q1 = 16.6
        self.q2 = 0.55
        self.q3 = 15.6
        self.q4 = 32.8
        self.A1 = 207

        self.simulation_time = simulation_time
        self.sample_time = sample_time

        """ constant input and disturbance"""
        # self.u = self.q3
        # self.d = self.q2

        """generate input and disturbance"""
        # np.random.seed(10)
        # self.U = np.random.rand(10) * 40
        # self.D = np.random.randn(10)
        self.U, self.D = input_generate.input_generate(self.simulation_time, [25, 35])
        plt.plot(range(self.simulation_time), self.U + self.D)
        plt.xlabel("time [s]")
        plt.ylabel("input u []")
        plt.show()

        """ initial condition"""
        self.x1_0, self.x2_0, self.x3_0 = self.W_a4, self.W_b4, self.h1
        self.S_0 = [self.x1_0, self.x2_0, self.x3_0]

        """time step"""
        self.t = np.linspace(0, self.simulation_time, self.simulation_time // self.sample_time)
        self.i = 0

    """state space equation"""
    def state_space(self, t, x):
        ipt = self.U[int(t // self.sample_time)]
        dis = self.D[int(t // self.sample_time)]

        dx1 = self.q1/(self.A1*x[2])*(self.W_a1-x[0]) + \
              1/(self.A1*x[2])*(self.W_a3-x[0])*ipt + \
              1/(self.A1*x[2])*(self.W_a2-x[0]) * dis

        dx2 = self.q1/(self.A1*x[2])*(self.W_b1-x[1]) + \
              1/(self.A1*x[2])*(self.W_b3-x[1])*ipt + \
              1/(self.A1*x[2])*(self.W_b2-x[1])*dis

        dx3 = 1/self.A1*(self.q1-self.C_v4*(x[2]+self.z) ** self.n) + 1/self.A1 * (ipt + dis)
        return [dx1, dx2, dx3]

    """solve ode with solve_ivp"""
    def ode_sovle(self):
        sol_2 = solve_ivp(self.state_space, t_span=(0, max(self.t)), y0=self.S_0, t_eval=self.t)
        self.state = sol_2.y
        return self.state


    """ constraint equation """
    def constraint(self, y):
        x1, x2, x3 = self.state[0, self.i], self.state[1, self.i], self.state[2, self.i]
        res = x1 + 10 ** (y - 14) + 10 ** (-y) + x2 * ((1 + 2*10 ** (y-self.pK2))/(1 + 10 ** (self.pK1 - y) + 10 ** (y - self.pK2)))
        return res

    """ calculate y [pH]"""
    def y_cal(self, s):
        y = [7.0]
        while self.i < len(self.t) - 1:
            y.append(fsolve(s, y[-1]))
            self.i += 1
        return y



if __name__ == '__main__':
    """set simulation time"""
    simulation_time = 25000

    ph = ph_model(simulation_time, 10)
    state = ph.ode_sovle()

    # plot state
    plt.subplot(311)
    plt.plot(ph.t, state[0])
    plt.subplot(312)
    plt.plot(ph.t, state[1])
    plt.subplot(313)
    plt.plot(ph.t, state[2])
    plt.show()

    # plot ph value
    y = ph.y_cal(ph.constraint)
    plt.plot(ph.t, y)
    plt.show()


"""
1. the relation between y and pH value.
2. input scale
"""
