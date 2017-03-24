import numpy as np
import argparse, textwrap
from support_functions import *
from write_nufft import *
#from plot_nufft import *



if __name__=='__main__':
    mainDir = '/hdd/work/wall-mounted-cube/hByDelta-0.6'

    fname = []
    fname.append(mainDir + '/medium-slip/postProcessing/force-left-side/0/force.dat')

    cols = [1, 7]
    uinf = 2.0
    h = 0.006

    
    description = textwrap.dedent(
        '''
        1-d NUFFT, method by Greengard et al.
        '''
        )

    parser = argparse.ArgumentParser(description = description,
                                     prefix_chars = '-')
    parser.add_argument('-write_nufft',
                        action='store_true',
                        default=False,
                        help='write 1d-nufft')
    parser.add_argument('-plot_nufft',
                        action='store_true',
                        default=False,
                        help='plot 1d-nufft')

    args = parser.parse_args()

    if args.write_nufft:
        write_nufft(fname, cols, uinf, h)
    if args.plot_nufft:
        plot_nufft(fname, cols, uinf, h)
