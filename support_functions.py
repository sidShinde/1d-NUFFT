import numpy as np
from math import pi
from math import exp
import re
from tqdm import tqdm


__all__ = ['fft_kernel_domain', 'add_contributions', 'get_reduced_time_signal', 'get_data']


def fft_kernel_domain(fTau, k, si, M, Mr):
    '''
    calculate the fft in kernel domain
    '''

    FTau = np.zeros( M, dtype=np.complex_ )

    for i in tqdm( range(M), ncols=100 ):
        kVector = np.exp( -1j*k[i]*si )
        FTau[i] = np.dot( fTau, kVector )
        FTau[i] = (1/Mr)*FTau[i]

    return FTau


def add_contributions(f, t, M, Mr, Msp, tau):
    '''
    spread the effect of the data point from given mesh
    to the neighbouring points on the over-sampled mesh
    '''

    m1 = np.arange(-Msp+1, Msp+1)

    temp = pi*m1/Mr
    temp = np.power(temp, 2)
    E3 = np.exp( -temp/tau )

    fTau = np.zeros(Mr)

    for i in tqdm( range(M), ncols=100 ):
        m, contribution = get_contribution(f[i], t[i], M, Mr, Msp, tau, E3, m1)

        if m < Msp:
            fTau[ 0:(Msp+m+1) ] += contribution[ (Msp-m-1):(2*Msp) ]
            fTau[ (Mr-(Msp-m-1)):Mr ] += contribution[ 0:(Msp-m-1) ]
        elif m+Msp >= Mr:
            fTau[ (m-Msp+1):Mr ] += contribution[ 0:(Msp+Mr-m-1) ]
            fTau[ 0:(m+Msp-Mr+1) ] += contribution[ (Msp+Mr-m-1):2*Msp ]
        else:
            fTau[ (m-Msp+1):(m+Msp+1) ] += contribution

    return fTau


def get_contribution(fj, xj, M, Mr, Msp, tau, E3, m1):
    '''
    return the contribution of fj to neighbouring Msp points
    '''

    p, m = choose_point(xj, Mr)
    E1 = exp( -(xj-p)**2/(4*tau) )
    E2 = exp( ( (xj-p)*pi )/(Mr*tau) )
    E2 = np.power( E2, m1 )

    contribution = fj*E1*E2*E3

    return m, contribution


def get_reduced_time_signal(x, f, uinf, h, ndtu=100):
    nonDimTime = h/uinf
    startTime  = x.min()
    endTime    = x.max()

    nNonDimTimeUnits = ( endTime - startTime ) / nonDimTime
    if nNonDimTimeUnits > ndtu:
        nNonDimTimeUnits = ndtu

    endTime = startTime + nNonDimTimeUnits*nonDimTime

    # index of the last data point:
    idx = np.argmin( x < endTime )

    # reduced arrays
    x = x[:idx]
    t = 2*pi*( x-startTime )/( endTime - startTime )

    f = f[:idx]

    return t, f, startTime, endTime


def choose_point(xj, Mr):
    '''
    return point on the over-sampled equi-spaced domain:
    '''

    for c in range(Mr-1, -1, -1):
        if 2*pi*(c/Mr) <= xj:
            p = 2*pi*(c/Mr)
            m = c
            break

    return p, m

def get_data(fname, skiprows=0):
    nCols = get_number_of_cols(fname, skiprows)
    data = [[] for i in range(nCols)]

    count = 0

    with open(fname) as f:
        for line in f:
            count += 1

            if count > skiprows:
                line = re.split(r'[(|)|\s]', line)

                # remove whitespaces from the line
                while '' in line:
                    line.remove('')

                try:
                    for i in range( nCols ):
                        data[i].append( float( line[i] ) )

                except:
                    continue

            # skip rows
            else: continue

    return np.array(data).T


def get_number_of_cols(fname, skiprows=0):
    count = 0
    with open(fname) as f:
        for line in f:
            count += 1

            if count > skiprows:
                line = re.split(r'[(|)|\s]', line)
                while '' in line:
                    line.remove('')

                try:
                    temp = float( line[0] )
                    nCols = len(line)
                    break
                except:
                    continue

            else: continue

    return nCols
