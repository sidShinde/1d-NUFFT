#! /usr/bin

import numpy as np
from math import pi
import re
from get_1d_nufft import *


__all__ = ['write_nufft', 'write_function', 'get_1d_nufft']


def write_nufft(fname, cols, uinf=1.0, h=1.0):
    for i in range( len(fname) ):
        print('\n case # ' + str(i+1) + ':')
        write_function(fname[i], cols, uinf, h)


def write_function(fname, cols, uinf=1.0, h=1.0):
    '''
    write k, Fk in the same directory as the input file
    '''

    print('\n reading data ...')
    data = get_data(fname)

    for i in range( len(cols) ):
        cols[i] = cols[i] - 1

    data = data[:, cols]

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
