import numpy as np
from math import pi
from support_functions import *


__all__ = ['get_1d_nufft']


def get_1d_nufft(x, f, uinf=1.0, h=1.0):
    '''
    x: sample points
    f: value at the sample points
    '''

    # sampled points mapped to [0, 2pi]
    t, f, startTime, endTime = get_reduced_time_signal(x, f, uinf, h)

    # number of sample points:
    N = t.shape[0]

    # define NUFFT parameters:
    M = N
    Mr = 2*M
    Msp = 12
    tau = Msp/(M**2)

    print('\n step 1: adding contributions ...')
    fTau = add_contributions(f, t, M, Mr, Msp, tau)

    print('\n step 2: calculating fourier coeffs in kernel domain ...')
    si = np.linspace(0, 2*pi, Mr)
    k = np.arange( -M/2, M/2 )
    FTau = fft_kernel_domain(fTau, k, si, M, Mr)

    print('\n step 3: calculating fourier coeffs ...')
    constant = np.sqrt(pi/tau)
    Fk = constant*np.exp( np.power( k, 2 )*tau )*FTau
    Fk = np.absolute(Fk)

    return k[ (int(M/2) + 1): ]/(endTime - startTime), Fk[ (int(M/2) + 1): ]
