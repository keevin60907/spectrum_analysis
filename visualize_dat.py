###############################
#     Copy Right @ YCC Lab    #
###############################

# Arthor: Tsung-Shan Yang
# Date  : 2018/ 08/ 08
# Usage : visualize helper from .dat to .png
# Exec  : python3 visualize_dat.py [dat_file] -m <0, 1, 2>

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc

parser = argparse.ArgumentParser(description='visualize helper from .dat to .png')
parser.add_argument('dat_file', help = 'enter the path of .dat file')
parser.add_argument('-m','--mode', type = int, default = 0, \
                     help = '0: print abs value spectrum\n \
                             1: print real value spectrum\n \
                             2: print img value spectrum')
args = parser.parse_args()

##########################
#    defualt settings    #
##########################

# the format should be: 
# x_coor1 y_coor1 real_part1 img_part1
# x_coor2 y_coor2 real_part2 img_part2
#  ...      ...      ...       ...
# x_coor (omega_1) from 0 ~ 1500 with inverval and n_datapoint
# y_coor (omega_2) from 0 ~ 1500

interval = 6.51
n_datapoint = 231

#######################
#    save settings    #
#######################

# pic: spectrum matrix
# pic_name: name of spectrum matrix
# defualt output: "visaulized_pic_name.png"

def save_pic(pic_name, pic):

    pic = np.flipud(pic)
    plt.matshow(pic)

    # set the visualize spectrum title as pic_name 
    plt.title(pic_name)

    # set x_axis
    plt.xlabel('$\omega_{1}$')
    plt.xticks([0, n_datapoint / 3, 2 * n_datapoint / 3, n_datapoint],\
        ['0', '500', '1000', '1500'])

    # set y_axis
    plt.ylabel('$\omega_{2}$')
    plt.yticks([0, n_datapoint / 3, 2 * n_datapoint / 3, n_datapoint],\
        ['1500', '1000', '500', '0'])

    # set the position of axis
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # add colorbar beside the graph
    plt.colorbar()

    # save the visualized spectrum
    plt.savefig('visualized_' + pic_name + '.png')
    plt.close()


def parse_dat(dat_file, mode):

    # read-in .dat file
    f = open(dat_file)
    pic = np.zeros((n_datapoint, n_datapoint))

    # parsing the data in the file
    for line in f.readlines():
        info = line.strip().split(' ')
        if len(info) != 1:
            x_coor = float(info[0])
            y_coor = float(info[1])
            real_part = float(info[2])
            img_part = float(info[3])

            # process the data points by different mode
            if mode == 0:
                pic[int(x_coor / interval)][int(y_coor / interval)] \
                    = real_part ** 2 + img_part ** 2
            elif mode == 1:
                pic[int(x_coor / interval)][int(y_coor / interval)] = real_part
            elif mode == 2:
                pic[int(x_coor / interval)][int(y_coor / interval)] = img_part

    # normalize the data in pixels
    base = pic.min()
    pic -= base
    pic = np.sqrt(pic)
    scalar = pic.max()
    pic /= scalar

    # save the data as .png file
    if mode == 0:
        save_pic(dat_file.split('.')[0] + '_abs', pic)
    elif mode == 1:
        save_pic(dat_file.split('.')[0] + '_real', pic)
    elif mode == 2:
        save_pic(dat_file.split('.')[0] + '_img', pic)

    return pic

if __name__ == '__main__':

    if args.mode not in [0, 1, 2]:
        print ('SORRY! Now we only support 3 modes QAQQQQQ')
    else:
        if os.path.isfile(args.dat_file):
            parse_dat(args.dat_file, args.mode)
        else:
            print ('OOPS! Please check whether the assigned file exists or not...')