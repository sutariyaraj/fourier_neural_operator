import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import random


def generate_heat_equation_data(amplitude=1.0, c=1, phase=0):
    """
    @param amplitude: maximum temperature of initial condition
    @param c: [int] no of harmonics of wave of initial condition
    @param phase: phase shift of the initial condition
    @return: numpy array
    """

    L=1             # length of rod in cm
    max_t = 2       # maximum time steps in seconds
    alpha = 0.03       # intrinsic property of metal used - fixed for same type of metal
    exponential_const = (c*np.pi/L)**2

    t = np.linspace(0, max_t, 200)
    x = np.linspace(0, L, 200)
    X, T = np.meshgrid(x, t)
    # X = X.reshape(-1,1)
    # T = T.reshape(-1,1)
    U = amplitude*(np.exp(-alpha*exponential_const*T))*np.sin((c*np.pi*X/L) + phase)
    U = U.reshape(200,200)

    # [DEBUG]
    # to see results
    # Domain bounds
    lb = np.array([0, 0.0])
    ub = np.array([L, max_t])
    def show_plot():
        plt.imshow(U.T, interpolation='nearest', cmap='YlGnBu',
                   extent=[lb[1], ub[1], lb[0], ub[0]],
                   origin='lower', aspect='auto')

        plt.ylabel('x (cm)')
        plt.xlabel('t (seconds)')
        plt.axis()
        plt.colorbar().set_label('Temperature (°C)')
        plt.show()
    # show_plot()
    return U.T


def generate_dataset():
    heat_dataset = np.zeros((1100, 200, 200))
    cnt = 0
    for amplitude in range(11):
        for harmonics in range(1,11):
            for phase_shift in range(10):
                phase_shift = random.random() * 2 * np.pi
                heat_dataset[cnt] = generate_heat_equation_data(amplitude=random.random(), c=harmonics, phase=phase_shift)
                cnt += 1
    np.random.shuffle(heat_dataset)
    return heat_dataset


if __name__ == '__main__':
    # generate range of data of heat equation
    dataset = generate_dataset()
    # save the matrix
    scipy.io.savemat('data/heat_N1100_T200_r200.mat', mdict={ 'u': dataset})