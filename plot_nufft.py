import numpy as np
import matplotlib
matplotlib.use('PDF')

__all__=['plot_nufft']


def plot_nufft(fname, uinf, h):

    data = []
    for i in range( len(fname) ):
        data.append( np.loadtxt(fname[i], delimiter=' ') )

    plot_function(data)
    

def plot_function(data):
    nCases = len( data )

    

    
        
