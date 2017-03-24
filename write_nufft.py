import numpy as np
from math import pi
import re
from support_functions import *


__all__ = ['write_nufft']


def write_nufft(fname, cols, uinf=1.0, h=1.0):
    for i in range( len(fname) ):
        print('\n case # ' + str(i+1) + ':')
        write_function(fname[i], cols, uinf, h)


def write_function(fname, cols, uinf=1.0, h=1.0):
    '''
    write k, Fk in the same directory as the input file
    '''
    
    print('\n reading data ...')
    data = read_data(fname, cols)

    # calculate 1d nufft:
    k, Fk = get_1d_nufft(data[:, 0], data[:, 1], uinf, h)

    print('\n writing fft ...')
    line = re.split(r'[/]', fname)
    del line[0]
    del line[-1]

    newFilePath = ''
    for word in line:
        newFilePath = newFilePath + '/' + word

    solution = np.array([k, Fk])
    solution = solution.T
    solFname = newFilePath + '/k-vs-Fk.csv'
    
    np.savetxt(solFname, solution, delimiter=' ', fmt='%.3e', newline='\r\n')


def get_1d_nufft(x, f, uinf=1.0, h=1.0):
    '''
    x: sample points
    f: value at the sample points
    '''
    
    # sampled points mapped to [0, 2pi]
    t, f = get_reduced_time_signal(x, f, uinf, h)

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

    return k[ (int(M/2) + 1): ], Fk[ (int(M/2) + 1): ] 


