import glob
import numpy as np
import argparse 
# from importlib.util import spec_from_file_location
import os
from tqdm import tqdm
from skimage.io import imread,imsave

from BaSic.utils import *
from BaSic.BaSic import *

import warnings



def main():
    warnings.filterwarnings("ignore")
    # parse arguments
    parser = argparse.ArgumentParser(description='Process images from source folder and save in destination folder')
    parser.add_argument('-lb','--light_background', help='Path of a python file containing four functions: image_io.py, pre_processing.py, processing.py,post_processing.py', required=False)
    parser.add_argument('-mw','--mean_width', help='Path of source folder with images', required=False)
    parser.add_argument('-nv','--noise_var', help='Path of source folder with images', required=False)    
    parser.add_argument('-s','--src', help='Path of source folder with images', required=True)
    parser.add_argument('-d','--dst', help='Path of destination folder', required=True)
    args = vars(parser.parse_args())

    if args['light_background'] == None:
        light_background = 'True'
        invert = True
    else:
        light_background = args['light_background']
        if light_background == 'True':
            invert = True
        else:
            invert = False

    if args['mean_width'] == None:
        mean_width = 10
    else:
        try:
            mean_width = float(args['mean_width'])
        except:
            print('Mean width not understood.')
    
    if args['src'] == None:
        print('No source images mentioned: example /path/to/src/folder/*.tif')
    else:
        src_folder = args['src']

    if args['dst'] == None:
        print('No destination directory mentioned: example /path/to/destination/*.tif')
    else:
        dst_folder = os.path.normpath(args['dst']) + os.path.sep
        # tmp = dst_folder.split('*')
        # save_format = tmp[1]
        # dst_folder = tmp[0]
    
    ## load model
    print('loading model ... ')
    basic = BaSiC()
    print('model ready.')    

    ## batch processing    
    flist = glob.glob(src_folder)
    for f in tqdm(flist):
        im = imread(f)        
        if len(im.shape)>2:
            im = im[:,:,0]
        sr,sz = im.shape
        scale = round(10.0/mean_width, 2)
        im = pre_processing(im,scale)
        im = basic.segment(im,invert)
        im = post_processing(im)
        im = resize(im,(sr,sc))
        im = (im>0.98)*255.0
        imsave(dst_folder + os.path.basename(f)[:-4] + '_basic.tif',im.astype(np.uint8))
    
