import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import _cntr as cntr

from scipy import optimize
import scipy.ndimage as ndimage
from sklearn.utils.extmath import randomized_svd
import cv2

from tp_utils import * 

def upsample_triggers(triggers,interval,rate):
    return np.linspace(
        triggers.min(),
        triggers.max() + interval,
        triggers.shape[0] * rate
    )

def znorm(data):
    return (data - data.mean())/data.std()

def interpolate_weights(data, triggers):
    data_interp = sp.interpolate.interp1d(
        data['Tracetimes0'].flatten(), 
        znorm(data['Traces0_raw'].flatten()),
        kind = 'linear'
    ) (triggers)
    
    return znorm(data_interp)

def lag_weights(weights,nLag):
    lagW = np.zeros([weights.shape[0],nLag])
    
    for iLag in range(nLag):
        lagW[iLag:-nLag+iLag,iLag] = weights[nLag:]/nLag
        
    return lagW

def extract_sta_rois(df_rois, roi_id, stimulus_path):
    
    data = df_rois.loc[roi_id]
    triggers = data['Triggertimes']
    
    weights = interpolate_weights(data, triggers)
    lagged_weights = lag_weights(weights, 5)

    stimulus = load_h5_data(stimulus_path)['k']
    stimulus = stimulus.reshape(15*20, -1)
    # stimulus = stimulus[:, 1500-len(weights):]
    # skip = np.floor(data['Triggertimes'][0]).astype(int)
    offset = 0
    stimulus = stimulus[:, offset:len(weights)+offset]

    sta = stimulus.dot(lagged_weights)
    U,S,Vt = randomized_svd(sta,3)
    
    return U[:, 0].reshape(15,20)

def extract_sta_soma(soma_data):
    
    triggers = soma_data['Triggertimes']
    triggers += -0.1
    weights = interpolate_weights(soma_data, triggers)
    lagged_weights = lag_weights(weights, 5)

    stimulus = load_h5_data('/home/Data/Ran/noise.h5')['k']
    stimulus = stimulus.reshape(15*20, -1)
    offset = 0
    stimulus = stimulus[:, offset:len(weights)+offset]

    sta = stimulus.dot(lagged_weights)
    U,S,Vt = randomized_svd(sta,3)
    
    return U[:, 0].reshape(15,20)

def pad_linestack(linestack, RF_resized, rec_center):
    
    linestack_xy = linestack.mean(2)
    
    Rx, Ry = np.array([RF_resized.shape[0]/2, RF_resized.shape[1]/2])
    Sx, Sy = rec_center[0]
    
    right_pad = np.round(Ry - (linestack_xy.shape[0] - Sy)).astype(int)
    left_pad = np.round(Ry - Sy).astype(int)
    top_pad = np.round(Rx - (linestack_xy.shape[0] - Sx)).astype(int)
    bottom_pad = np.round(Rx - Sx).astype(int)
    
    linestack_padded = np.pad(linestack_xy, ((top_pad, bottom_pad), (left_pad, right_pad)), 'constant')
    
    return linestack_padded

def get_contour(RF, threshold=2.5, sigma=(1,1)):

    RF = ndimage.gaussian_filter(RF, sigma=sigma, order=0)
    x, y = np.mgrid[:RF.shape[0], :RF.shape[1]]
    c = cntr.Cntr(x,y, RF)

    res = c.trace(RF.mean()+ RF.std() * threshold)
    # res = c.trace(RF.max() - threshold * RF.std())
    return (res[0][:, 0], res[0][:, 1]), RF

def resize_RF(RF, RF_pixel_size, stack_pixel_size):
    scale_factor = RF_pixel_size/stack_pixel_size
    return sp.misc.imresize(RF, size=scale_factor, interp='bicubic')