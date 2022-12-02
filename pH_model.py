# -*- coding: utf-8 -*- 
# @Time : 2022/11/17 14:55 
# @Author : Yinan 
# @File : pH_model.py
from scipy.integrate import solve_bvp, solve_ivp, odeint
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import input_generate


class PHSimulator:
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
        self.W_a3 = -3.05e-3
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

        """generate input"""
        self.U = input_generate.input_generate(self.simulation_time, [25, 35])

        """ initial condition"""
        self.x1_0, self.x2_0, self.x3_0 = self.W_a4, self.W_b4, self.h1
        self.S_0 = [self.x1_0, self.x2_0, self.x3_0]

        """time step"""
        self.t = np.linspace(0, self.simulation_time, self.simulation_time // self.sample_time)
        self.i = 0

    """state space equation"""
    def state_space(self, t, x):
        ipt = self.U[int(t-0.1)]
        dis = self.q2

        dx1 = self.q1/(self.A1 * x[2]) * (self.W_a1 - x[0]) + \
              1 / (self.A1 * x[2]) * (self.W_a3-x[0]) * ipt + \
              1 / (self.A1 * x[2]) * (self.W_a2-x[0]) * dis

        dx2 = self.q1/(self.A1 * x[2]) * (self.W_b1-x[1]) + \
              1 / (self.A1 * x[2])*(self.W_b3-x[1]) * ipt + \
              1 / (self.A1 * x[2])*(self.W_b2-x[1]) * dis

        dx3 = 1 / self.A1 * (self.q1 - self.C_v4 * (x[2]+self.z) ** self.n) + 1/self.A1 * (ipt + dis)
        return [dx1, dx2, dx3]

    """solve ode with solve_ivp"""
    def ode_solve(self):
        sol_2 = solve_ivp(self.state_space, t_span=(0, max(self.t)), y0=self.S_0, t_eval=self.t)
        self.state = sol_2.y
        return self.state

    """ constraint equation """
    def constraint(self, y):
        x1, x2, x3 = self.state[:, self.i]
        return x1 + 10 ** (y - 14) + 10 ** (-y) + x2 * ((1 + 2*10 ** (y-self.pK2))/(1 + 10 ** (self.pK1 - y) + 10 ** (y - self.pK2)))

    """ calculate y [pH]"""
    def y_cal(self, s):
        self.y = [7.]
        while self.i < len(self.t) - 1:
            self.y.append(fsolve(s, self.y[-1]))
            self.i += 1
        return self.y


def main():
    # TODO: Add whitenoise to input and output data to prevent overfitting.
    #   Keep the unnoised data for plotting however.
    np.random.seed(100)

    training_samples = 4400
    sampling_time = 10

    simulation_time = sampling_time * training_samples

    ph_simulator = PHSimulator(simulation_time, sampling_time)
    _ = ph_simulator.ode_solve()
    ph_calc = np.array(ph_simulator.y_cal(ph_simulator.constraint))

    u_train = ph_simulator.U + np.random.normal(0, 0.01, size=ph_simulator.U.shape)
    print('u_train', u_train.shape)
    y_train = ph_calc + np.random.normal(0, 0.01, size=ph_calc.shape)
    print('y_train', y_train.shape)

    u_mean = np.mean(u_train)
    u_dev = (np.max(u_train) - np.min(u_train)) / 2

    def normalize(u):
        return (u - u_mean) / u_dev

    u_train_normed = normalize(u_train)

    fig, ax = plt.subplots(2, sharex='all')
    ax[0].plot(
        range(ph_simulator.simulation_time),
        np.hstack(u_train_normed),
        color='C0'
    )
    ax[1].plot(ph_simulator.t, np.hstack(y_train), color='C1')

    ax[0].set_ylabel('Input')
    ax[1].set_ylabel('PH')
    fig.suptitle('States')
    ax[-1].set_xlabel('time [s]')

    plt.show()


if __name__ == '__main__':
    main()
