import numpy as np
from skimage.transform import resize,rescale
# image processing stuff
from skimage.feature import shape_index
from skimage.util import pad,random_noise
from utils import *
from tensorflow.keras.models import load_model
class Misic():
    def __init__(self,model_name):
        self.size = 256
        self.model = load_model(model_name)
        #self.model.compile()
    
    # The preprocessing function that converts image to shape index at 3 scales
    # The preprocessing function that converts image to shape index at 3 scales
    def preprocess(self,im):
        sh = np.zeros((im.shape[0],im.shape[1],2))
        if np.max(im) ==0:
            return sh
        pw = 8
        im = np.pad(im,((pw,pw),(pw,pw)),'reflect')
        sh = np.zeros((im.shape[0],im.shape[1],2))    
        sh[:,:,0] = shape_index(im,0)
        sh[:,:,1] = shape_index(im,0.5)
        #sh[:,:,2] = shape_index(im,1.0)
        #sh = 0.5*(sh+1.0)
        return sh[pw:-pw,pw:-pw,:]
    
    def segment(self,im,invert = False):
        im = normalize2max(im)        
        pw = 16
        if invert:
            im = 1.0-im
        im = np.pad(im,pw,'reflect')
        sh = self.preprocess(im)
        
        tiles,params = extract_tiles(sh,size = self.size,padding = 16)
        yp = self.model(tiles)

        return stitch_tiles(yp,params)[pw:-pw,pw:-pw,:]    
