import os
import h5py
import glob
import pickle
import skfmm
import scipy.ndimage
import scipy.misc

import numpy as np
import tifffile as tiff
import pandas as pd

from skimage.feature import match_template

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_scalebar.scalebar import ScaleBar

from matplotlib import rcParams
rcParams['font.family'] = 'DejaVu Sans'


def get_data_paths(experimenter, expdate):


    rootdir = '/gpfs01/euler/data/Data/' + experimenter + '/' + expdate 

    if len(os.listdir(rootdir)) > 1:
        exp_num = input('Which experiment? [{}]: '.format(os.listdir(rootdir)))
    else:
        exp_num = os.listdir(rootdir)[0]

    celldir = rootdir + '/' + exp_num

    for file in os.listdir(celldir + '/Raw'):
        if ('traces' in file.lower()):
            trace_path = celldir + '/Raw/' + file

    for file in os.listdir(celldir + '/Pre'):
        if ('stack.h5' in file.lower()):
            stack_h5_path = celldir + '/Pre/' + file 
        if ('_s_d' in file.lower()):
            soma_h5_path = celldir + '/Pre/' + file

    try:
        soma_h5_path
    except NameError:
        print("There's no SMP_CX_S_DnoiseGC30.h5, try using SMP_CX_S_Chirp.h5")
        for file in os.listdir(celldir + '/Pre'):
            if ('_s_c' in file.lower()):
                soma_h5_path = celldir + '/Pre/' + file

    dendrites_h5_paths = []
    for file in os.listdir(celldir + '/Pre'):
        if ('_d' in file.lower() and 'chirp' not in file.lower() and 's_d' not in file.lower()):
            dendrites_h5_paths.append(celldir + '/Pre/' + file)
    dendrites_h5_paths.sort()

    print('\ntrace_path: ', '\n' + trace_path + '\n')
    print('soma_h5_path: ', '\n' + soma_h5_path + '\n')
    print('stack_h5_path: ',  '\n' + stack_h5_path + '\n')

    for dfile_idx, dfile in enumerate(dendrites_h5_paths):
        print('dendrites_h5 [{}]: '.format(dfile_idx), '\n' + dfile + '\n')

    return {'trace_path': trace_path,
            'soma_h5_path': soma_h5_path,
            'stack_h5_path': stack_h5_path,
            'dendrites_h5_paths': dendrites_h5_paths}, celldir, exp_num

#######################
## Helper functions ##
######################

def load_h5_data(file_name):
    with h5py.File(file_name,'r') as f:
        return {key:f[key][:] for key in list(f.keys())}

def read_trace(trace_path):
    
    '''
    Read `.trace` file into pandas DataFrame.

    Parameter
    =========
    trace_path: string
        - the path of .trace file

    Return
    ======
    df: pandas DataFrame
    
    Example
    =======
    
        trace_path = '/home/Data/Ran/C0/C0_stack_wDataCh0.traces'
        df = read_trace(trace_path)
    
    Reference
    =========
    http://stackoverflow.com/a/28267291/2135095
    
    '''
    
    import xml.etree.ElementTree as ET
    import gzip

    def trace2xml(trace_path):

        trace_name = trace_path.split('/')[-1].split('.')[-2]

        with gzip.open(trace_path, 'rb') as f:
            xml_binary = f.read()

        return xml_binary.decode('utf-8')

    def iter_points(path):
        path_attr = path.attrib
        for point in path.iterfind('.//point'):
            point_dict = path_attr.copy()
            point_dict.update(point.attrib)
    #         doc_dict['data'] = doc.text
            yield point_dict

    def iter_paths(etree):
        for path in etree.iterfind('.//path'):
            for row in iter_points(path):
                yield row

    xml_data = trace2xml(trace_path)
    etree = ET.fromstring(xml_data) #create an ElementTree object 
    
    df = pd.DataFrame(list(iter_paths(etree)))
    
    imagesize = etree.find('imagesize').attrib
    samplespacing = etree.find('samplespacing').attrib
    
    meta_info = {}
    meta_info.update(imagesize)
    meta_info.update(samplespacing)
    
    return df, meta_info

def trace2linestack(df, meta_info):
    
    stack_size = [int(meta_info['width']), 
                  int(meta_info['height']),
                  int(meta_info['depth'])]
    
    x = [int(i) for i in df.x.tolist()]
    y = [int(i) for i in df.y.tolist()]
    z = [int(i) for i in df.z.tolist()]
    
    coords = np.vstack((np.vstack((x,y)), z)).T
    
    linestack = np.zeros(stack_size)
    for c in coords:
        linestack[tuple(c)] = 1
        
    return linestack

#################################
## Get soma info from stack.h5 ##
#################################


def get_info_soma(stack, adjust_factor=1):
    """
    Automatic detection of soma centroid, radius and mask from stack data. 

    Parameter
    =========
    stack: array-like (512, 512, Z)
        - stack data from SMP_Cx_Stack.h5. 
    
    Return
    ======
    soma: dict
        - a dict contains centroid, radius and mask of the soma. 

    Example
    =======
        In [1]: soma = get_info_soma(stack['wDataCh0'])
        
        In [2]: soma['centroid']
        Out[2]: array([255, 256,  23])
        
        In [3]: soma['radius']
        Out[3]: 8.6029908211272286

    Reference
    ========= 
        https://github.com/RivuletStudio/rivuletpy/blob/master/rivuletpy/soma.py
    """

    threshold = np.mean(stack) + adjust_factor * np.std(stack)
    bimg = (stack > threshold).astype('int')    
    
    dt = skfmm.distance(bimg, dx=1.1)  # Boundary DT

    radius = dt.max()
    centroid = np.asarray(np.unravel_index(dt.argmax(), dt.shape))

    ballvolume = np.zeros(bimg.shape)
    ballvolume[centroid[0], centroid[1], centroid[2]] = 1
    stt = scipy.ndimage.morphology.generate_binary_structure(3, 1)
    for i in range(np.ceil(radius * 2.5).astype(int)):
        ballvolume = scipy.ndimage.binary_dilation(ballvolume, structure=stt)

    mask_3d = np.logical_and(ballvolume, bimg)

    mask_xy = mask_3d.sum(2)
    mask_xy[mask_xy !=0] = 1
    mask_xy = np.ma.masked_array(mask_xy, ~mask_3d.any(2))
    
    mask_xz = mask_3d.sum(1)
    mask_xz[mask_xz !=0] = 1
    mask_xz = np.ma.masked_array(mask_xz, ~mask_3d.any(1))
    
    mask_yz = mask_3d.sum(0)
    mask_yz[mask_yz !=0] = 1
    mask_yz = np.ma.masked_array(mask_yz, ~mask_3d.any(0))
    
    soma = {'centroid': centroid, 
            'radius': radius, 
            'mask_xy': mask_xy,
            'mask_xz': mask_xz,
            'mask_yz': mask_yz,
            'mask_3d': mask_3d}

    return soma

# def plot_linestack(df_trace, meta_trace, info_soma, figsize=(16,16), savefig=False, save_path='.'):

#     linestack = trace2linestack(df_trace, meta_trace)

#     plt.figure(figsize=figsize)
#     plt.imshow(linestack.sum(2), origin='lower', cmap=plt.cm.binary)
#     plt.imshow(info_soma['mask_2d'], origin='lower', cmap=plt.cm.binary, vmin=0.0, alpha=0.3)

#     plt.grid('off')

def get_pixel_size_rec(rec, verbose=False):
    
    """
    Return the real length (in um) of each pixel point.
    """
    len_rec_x_pixel = 64
    len_rec_x_um = 71.5 / rec['wParamsNum'][30]
    
    rec_pixel_size = len_rec_x_um / len_rec_x_pixel
    
    if verbose:
        print("the real length of each pixel in this recording is: \n{0} um".format(rec_pixel_size))
    
    return rec_pixel_size


def get_pixel_size_stack(stack, verbose=False):
    
    """
    Return the real length (in um) of each pixel point.
    """
    len_stack_x_pixel = 512
    len_stack_x_um = 71.5 / stack['wParamsNum'][30]
    
    stack_pixel_size = len_stack_x_um / len_stack_x_pixel
    
    if verbose:
        print("the real length of each pixel in stack image is: \n{0} um".format(stack_pixel_size))
    
    return stack_pixel_size

def get_scale_factor(rec, stack):
    
    """
    get the scale factor from rec to stack, 
    e.g. scipy.misc.imresize(rec, size=scale_factor, interp='nearest')
    would make the rec into the same scale as stack. 
    """
    
    rec_pixel_size = get_pixel_size_rec(rec)
    stack_pixel_size = get_pixel_size_stack(stack)
    
    return rec_pixel_size / stack_pixel_size

def resize_roi(rec, stack):
    return scipy.misc.imresize(rec['ROIs'], size=get_scale_factor(rec, stack), interp='nearest')

def resize_rec(rec, stack):
    
    reci = rec_preprop(rec)
    
    return scipy.misc.imresize(reci, size=get_scale_factor(rec, stack), interp='nearest')

def rotate_rec(rec, stack,angle_adjust=0):
    
    ang_deg = rec['wParamsNum'][31] + angle_adjust# ratoate angle (degree)
    ang_rad = ang_deg * np.pi / 180 # ratoate angle (radian)
    
    rec_rec = resize_rec(rec, stack)
    rec_rot = scipy.ndimage.interpolation.rotate(rec_rec, ang_deg)
    
    (shift_x, shift_y) = 0.5 * (np.array(rec_rot.shape) - np.array(rec_rec.shape))
    (cx, cy) = 0.5 * np.array(rec_rec.shape)
    
    px, py = (0, 0) # origin
    px -= cx
    py -= cy
    
    xn = px * np.cos(ang_rad) - py*np.sin(ang_rad)
    yn = px * np.sin(ang_rad) + py*np.cos(ang_rad)
    
    xn += (cx + shift_x)
    yn += (cy + shift_y)    
    
    # the shifted origin after rotation
    
    return rec_rot, (xn, yn)

def rec_preprop(rec):
    
    reci = rec['wDataCh0'].mean(2)
    reci[:4, :] = reci.mean() - 0.5*reci.std()
    
    return reci

def rotate_roi(rec, stack, angle_adjust=0):
    
    ang_deg = rec['wParamsNum'][31] + angle_adjust # ratoate angle (degree)
    ang_rad = ang_deg * np.pi / 180 # ratoate angle (radian)
    
    rec_rois = resize_roi(rec, stack)
    rec_rois_rot = scipy.ndimage.interpolation.rotate(rec_rois, ang_deg, cval=255, order=0)
    rec_rois_rot = np.ma.masked_where(rec_rois_rot == 255, rec_rois_rot)

    (shift_x, shift_y) = 0.5 * (np.array(rec_rois_rot.shape) - np.array(rec_rois.shape))
    (cx, cy) = 0.5 * np.array(rec_rois.shape)
    
    labels = np.unique(rec_rois)[:-1][::-1]
    # reverse the lables to keep consistent with the labels of raw traces

    px = [np.vstack(np.where(rec_rois == i)).T[:, 0].mean() for i in labels] 
    py = [np.vstack(np.where(rec_rois == i)).T[:, 1].mean() for i in labels]

    px -= cx
    py -= cy
    
    xn = px * np.cos(ang_rad) - py*np.sin(ang_rad)
    yn = px * np.sin(ang_rad) + py*np.cos(ang_rad)
    
    xn += (cx + shift_x)
    yn += (cy + shift_y)
    
    return rec_rois_rot, np.vstack([xn, yn]).T

def rel_position_um(soma, d):
    
    """
    Relative position between dendrites and soma in um.
    
    Return
    ======
    
    array([YCoord_um, XCoord_um, ZCoord_um])
    """
    
    return soma['wParamsNum'][26:29] - d['wParamsNum'][26:29]

def roi_matching(image, template):
    
    result = match_template(image, template)
    ij = np.unravel_index(np.argmax(result), result.shape)
    
    return np.array(ij)


def point_on_which_path(df_trace, point_coord, verbose=False):

    '''
    Get the id of a path which a ROI is on.
    
    Paramenters
    ===========
    df_trace: pandas DataFrame
        - DataFrame of the trace.
    point_coord:
        the xyz-coordinate of a point
    
    Return
    ======
    path_id: str
        the id of a path
    '''
    
    path_list = np.unique(df_trace['id'].tolist())
    
    for path_id in path_list:
        
        path_coords = get_path_coords(df_trace, path_id)
        if (point_coord == path_coords).all(1).any():
            if verbose:
                print('ROI {} is on path {}'.format(point_coord, path_id))
            return int(path_id)

def get_df_paths(df_trace):
    
    all_paths = get_all_paths(df_trace)
    df_paths = pd.DataFrame(list(all_paths.items()), columns=['path_id', 'path'])

    df_paths.sort_values(['path_id'], ascending=[True], inplace=True)
    df_paths.index = df_paths.path_id.as_matrix()

    return df_paths

def get_all_paths(df_trace):
    
    path_dict = {}
    path_list = np.unique(df_trace['id'].tolist())
    
    for path_id in path_list:
        path_dict[int(path_id)] = get_path_coords(df_trace, path_id)
    
    return path_dict

def get_path_coords(df_trace, path_id):

    '''
    Get the coordinates of a trace/path.

    Parameters
    ==========
    
    df_trace: pandas DataFrame
        - DataFrame of the trace.
    path_id: int
        - ID of the path

    * Here we call each "line" of the cell trace as Path,
    * be consistent with the term used in ImageJ.

    Return
    ======

    coords: array-like
        - the coordinates of one path. 
    '''
    
    x = df_trace[df_trace['id'] == str(path_id)]['x'].astype('int').tolist()
    y = df_trace[df_trace['id'] == str(path_id)]['y'].astype('int').tolist()
    z = df_trace[df_trace['id'] == str(path_id)]['z'].astype('int').tolist()
    coords = np.vstack([[x,y], z]).T
    
    return coords

def get_distance(point_a, point_b):
    
    """
    Euclidean distance between two points.
    """
    
    return np.sqrt(np.sum((point_a - point_b) ** 2, 1))


def connected_with_soma(all_paths, key, soma_info, threshold=5):
    
    path = all_paths[key]
    
    xm, ym, zm = np.ma.where(soma_info['mask_3d'])
    soma_shape = np.array([xm, ym, zm]).T
    
    if np.min(get_distance(path[0], soma_shape)) < threshold or np.min(get_distance(path[1], soma_shape)) < threshold:
        return True
    else:
        return False

def get_connect_to(all_paths, key, exception=None):
    
    path = all_paths[key]
    
    sub_paths = all_paths.copy()
    sub_paths.pop(key)

    if exception != None:
        paths_to_delete = exception[key]
        for path_to_delte in paths_to_delete:
            sub_paths.pop(path_to_delte)
    
    def get_target(point, paths):
        # get shortest pair distance between the point of start path and all other paths.
        pair_distances = []
        for path_id in paths.keys():
            shortest_distance = np.min(get_distance(point, paths[path_id]))
            pair_distances.append([path_id, shortest_distance])
        pair_distances = np.array(pair_distances)

        target_path_id = pair_distances[np.argmin(pair_distances[:, 1])][0].astype(int)
        distance_to_target_path = np.min(pair_distances[:, 1])
    
        return (target_path_id, distance_to_target_path)
    
    target = get_target(path[0],  sub_paths)
    target_path_id = target[0]
    
    target_point = sub_paths[target_path_id][np.argmin(get_distance(path[0], sub_paths[target_path_id]))]
        
    return target_path_id, target_point


def distance_point2end(point, path_id, df_paths):
    
    path = df_paths.loc[path_id].path_updated
    point_loc = np.where((point  == path).all(1))[0]
    
    if len(point_loc) > 1:
        point_loc = point_loc[0]
        
    # segment = path[:int(point_loc), :] 
    segment = path[:int(point_loc), :] # should include the point_loc into the segment?
    segment_length = np.sum(np.sqrt(np.sum((segment[1:] - segment[:-1])**2, 1)))    
    
    return segment_length, segment

def get_segment_and_branchpoints(connected_points, point_of_interest, path_id, df_paths):
    
    if len(connected_points) != 0:
        
        points_on_path = np.vstack([point_of_interest, connected_points])
    
        results =[distance_point2end(point, path_id, df_paths) for point in points_on_path]
        # results =[(segment_length0, segment0), (segment_length1, segment1), ...]
        distances = np.array([results[i][0] for i in range(len(results))]) # lengths of all points to the end of the path

        segment = results[0][1] # the segment from poi to the end of the path 
        segment_length = distances[0]

        # new added (need more time to review)
        change_order = np.argsort(distances)[::-1]

        distances_sorted = distances[change_order]
        points_on_path_sorted = points_on_path[change_order]
        
        if sum(segment_length > distances_sorted) != 0:
            branchpoints = points_on_path_sorted[segment_length > distances_sorted]        
        else:
            branchpoints = []
    else:
        
        point_on_path = point_of_interest
        results = distance_point2end(point_on_path, path_id, df_paths)
        segment  = results[1]
        branchpoints = []
        
    return segment, branchpoints


def get_all_segments_and_branchpoints(roi_coordinate, path_id, df_paths):

    connected_points = df_paths.loc[path_id].connected_by_at
    
    segment, branchpoints  = get_segment_and_branchpoints(connected_points, roi_coordinate, path_id, df_paths)
    
    all_segments = [segment]
    all_branchpoints = [branchpoints]

    i = 0
    while df_paths.loc[path_id].connect_to != -1:
        
        point_of_interest = df_paths.loc[path_id].connect_to_at
        path_id = df_paths.loc[path_id].connect_to
        
        connected_points = df_paths.loc[path_id].connected_by_at
        
        segment, branchpoints = get_segment_and_branchpoints(connected_points, point_of_interest, path_id, df_paths)
        
        if len(branchpoints) != 0:
            branchpoints = np.vstack([point_of_interest, branchpoints])
        else: 
            branchpoints = point_of_interest
                
        all_segments.append(segment)    
        all_branchpoints.append(branchpoints)
        
        # if i > 100:
        #     print('Cannot find paths back to soma, stop!')
        #     all_segments = []
        #     all_branchpoints = [] 
        #     continue

    return all_segments, all_branchpoints


#########################
## Pairwise Statistics ##
#########################

def unique_row(a):
    
    b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
    _, idx = np.unique(b, return_index=True)
    
    unique_a = a[idx]
    
    return unique_a

def get_segment_from_roi_to_nexus_pt(sm_roi_to_soma, nexus_pt):    
    
    nexus_pt_loc = np.where((nexus_pt == sm_roi_to_soma).all(1))[0][0]

    return sm_roi_to_soma[nexus_pt_loc:]

def get_info_roi2roi(df_trace, df_paths, df_rois, info_soma, roi_id0, roi_id1):
    
    roi0 = df_rois.loc[roi_id0].roi_coords
    roi1 = df_rois.loc[roi_id1].roi_coords

    bpts0 = df_rois.loc[roi_id0].branchpoints # the first point is always the soma centroid
    bpts1 = df_rois.loc[roi_id1].branchpoints

    overlap_pts = [pt0 for pt0 in bpts0[:-1] for pt1 in bpts1[:-1] if (pt0 == pt1).all()] # we don't count the soma centroid here
    
    sm_roi0_to_soma = df_rois.loc[roi_id0].segments
    sm_roi1_to_soma = df_rois.loc[roi_id1].segments

    if len(overlap_pts)>0: # not intersected

        nexus_pt = overlap_pts[0] # nexus point other than the soma
        
        nexus_loc_in_bpts0 = np.where([(nexus_pt == pt).all() for pt in bpts0])[0][0]
        nexus_loc_in_bpts1 = np.where([(nexus_pt == pt).all() for pt in bpts1])[0][0]

        sm0 = get_segment_from_roi_to_nexus_pt(sm_roi0_to_soma, nexus_pt)
        sm1 = get_segment_from_roi_to_nexus_pt(sm_roi1_to_soma, nexus_pt) 

        if sm0.shape[0] > sm1.shape[0]:
            A = sm0.copy() # A is the longer one
            B = sm1.copy() # B is the shorter one
        else:
            A = sm1.copy()
            B = sm0.copy()

        loc = np.where((B[-1] == A).all(1))[0] # loc of shorter roi on longer roi

        if len(loc) == 0: # the first point of B is not on A, then they are branched out.
            sms_between = [sm0, sm1]
            bpts_between = np.vstack([bpts0[1:nexus_loc_in_bpts0+1], bpts1[1:nexus_loc_in_bpts1+1]])
        else: # B is within A
            sms_between = [A[loc[0]:]]
            bpts_between = np.vstack([bpts0[1:nexus_loc_in_bpts0], bpts1[1:nexus_loc_in_bpts1]])

    else: 

        if (roi0 == sm_roi1_to_soma).all(1).any():

            loc = np.where((roi0 == sm_roi1_to_soma).all(1))[0]
            sms_between = [sm_roi1_to_soma[loc[0]:]]
            bpts_between = bpts1[1:]

        elif (roi1 == sm_roi0_to_soma).all(1).any():

            loc = np.where((roi1 == sm_roi0_to_soma).all(1))[0]
            sms_between = [sm_roi0_to_soma[loc[0]:]]
            bpts_between = bpts0[1:]

        else:
            sms_between = [sm_roi0_to_soma, sm_roi1_to_soma]
            bpts_between = np.vstack([bpts0, bpts1])

    bpts_between = unique_row(bpts_between)

    return sms_between, bpts_between

#####

def get_df_branchpoints(df_trace, df_rois, df_paths, stack_pixel_size, info_soma):
    
    branchpoints_tmp = df_rois.branchpoints.values
    branchpoints_tmp = np.array([bpt for bpt in branchpoints_tmp if bpt != np.array([])])

    tmp = np.vstack(branchpoints_tmp)
    
    all_branchpoints = unique_row(branchpoints_tmp)
    all_branchpoints = np.array([bpt for bpt in all_branchpoints if bpt != info_soma['centroid']])

    df_branchpoints = pd.DataFrame(columns=('branchpoint_id',
                                            'branchpoint',
                                            'branchpoint_order'))
    idx = 0
    for row in t.df_rois.iterrows():
        
        i = row[0]
        bpts = row[1]['branchpoints'][::-1]
        
        
        for j, bpt in enumerate(bpts):
            
            bpts_saved = df_branchpoints.branchpoint.tolist()
            
            if len(bpts_saved) == 0:
            
                if (bpt == bpts_saved):
                    continue
                else:
                    df_branchpoints.loc[idx] = [idx, bpt, j]
                    idx +=1
            else:
                if (bpt == bpts_saved).all(1).any():
                    continue
                else:
                    df_branchpoints.loc[idx] = [idx, bpt, j]
                    idx +=1
                
    return df_branchpoints


## get RF overlap
import cv2
from shapely.geometry import Polygon

def point_distance(pt0, pt1):
    if pt1.shape != (2,):
        return np.sqrt(np.sum((pt0 - pt1) ** 2, 1))
    else:
        return np.sqrt(np.sum((pt0 - pt1) ** 2))

def check_cntr_interception(sCntr, bCntr):
    
    sCntrPoly = Polygon(sCntr)
    bCntrPoly = Polygon(bCntr)
    
    return sCntrPoly.intersects(bCntrPoly)

def interpolate_cntr(cntr, n=10):
    
    if n > 2:
    
        x = cntr[:, 0]
        y = cntr[:, 1]

        x_intep = []
        for i in range(len(x)-1):
            x_intep.append(np.linspace(x[i], x[i+1], n))
        x_intep = np.hstack(x_intep)   

        y_intep = []
        for j in range(len(y)-1):
            y_intep.append(np.linspace(y[j], y[j+1], n))
        y_intep = np.hstack(y_intep)

        return np.vstack([x_intep, y_intep]).T
    elif n <=2:
        return cntr



def get_inner_cntr(cntr0, cntr1, n, stack_pixel_size): 

    cntr0 = interpolate_cntr(cntr0, n)
    cntr1 = interpolate_cntr(cntr1, n)
    
#     print(cntr0.shape)
    
    if cntr0.shape[0] > cntr1.shape[0]:
        sCntr = cntr1.copy()
        bCntr = cntr0.copy()
    else:
        sCntr = cntr0.copy()
        bCntr = cntr1.copy()

    if check_cntr_interception(sCntr, bCntr):
    
        center = (sCntr.mean(0) + bCntr.mean(0) ) / 2

        num_sCntr_pt = sCntr.shape[0]

        inner_pts = []

        for sIdx in np.arange(num_sCntr_pt):

            sCntr_pt = sCntr[sIdx]

            bIdx = np.argmin(point_distance(sCntr_pt, bCntr))
            bCntr_pt = bCntr[bIdx]
            
            sCntr_pt2center = point_distance(sCntr_pt, center)
            bCntr_pt2center = point_distance(bCntr_pt, center)

            if  sCntr_pt2center > bCntr_pt2center:
                inner_pt = bCntr_pt
                outer_pt = sCntr_pt
            else:
                inner_pt = sCntr_pt
                outer_pt = bCntr_pt
            

            if inner_pts != []:

                if not any((inner_pt == pt).all() for pt in inner_pts):

                    btw_dis = point_distance(inner_pt, inner_pts[-1])

                    if btw_dis > sCntr_pt2center: # if btw_dis is larger then the sCntr_pt2center,
                                                  # then it means the inner_pt is wrong, should be replaced by outer_pt  
                        inner_pts.append(outer_pt)
                    else:
                        inner_pts.append(inner_pt)

            else:
#                 print(sIdx, len(inner_pts) ,inner_pt)
                inner_pts.append(inner_pt)
    
        overlap_cntr = np.vstack(inner_pts)

        btw_dis = np.sqrt(((overlap_cntr[:-1] - overlap_cntr[1:])**2).sum(1))
        points_to_delete = np.where(btw_dis>10)[0]
        num_points_to_delete = len(points_to_delete)

        if num_points_to_delete > 2:
            if np.mod(num_points_to_delete, 2) == 1:
                points_to_delete = points_to_delete[:-1]

            points_to_delete = points_to_delete.reshape(int(num_points_to_delete/2) , 2)
            for idx_pair_of_points in range(points_to_delete.shape[0]):
                start_point = points_to_delete[idx_pair_of_points][0] + 1
                end_point = points_to_delete[idx_pair_of_points][1] + 1
                # print(start_point, end_point)

                if len(range(start_point,end_point))>20:
                    overlap_cntr = np.delete(overlap_cntr, [start_point,end_point], axis=0)
                else:
                    overlap_cntr = np.delete(overlap_cntr, range(start_point,end_point), axis=0)
        elif num_points_to_delete==2:
            start_point = points_to_delete[0] + 1
            end_point = points_to_delete[1] + 1
            if len(range(start_point,end_point))>20:
                overlap_cntr = np.delete(overlap_cntr, [start_point,end_point], axis=0)
            else:
                overlap_cntr = np.delete(overlap_cntr, range(start_point,end_point), axis=0)
        else:
            overlap_cntr = overlap_cntr
        
        overlap_size = cv2.contourArea(overlap_cntr.astype(np.float32)) * (stack_pixel_size * stack_pixel_size)/1000 
        
    else:
        overlap_cntr = []
        overlap_size = 0

    cntr0_size = cv2.contourArea(cntr0.astype(np.float32)) * (stack_pixel_size * stack_pixel_size)/1000 
    cntr1_size = cv2.contourArea(cntr1.astype(np.float32)) * (stack_pixel_size * stack_pixel_size)/1000 
    
    if cntr0_size > cntr1_size:
        bigger_size = cntr0_size
        smaller_size = cntr1_size
    else:
        bigger_size = cntr1_size
        smaller_size = cntr0_size        

    return overlap_cntr, {'overlap_size': overlap_size,
                          'cntr0_size': cntr0_size,
                          'cntr1_size': cntr1_size,
                          'bigger_size': bigger_size,
                          'smaller_size': smaller_size}
        
##########################################
## calculate the angle between two rois ##
##########################################

def get_angle_roi2roi(df_rois, info_soma, roi_id0, roi_id1, dim=2):
    
    if dim == 2:
        v0 = df_rois.loc[roi_id0].roi_coords_stack_xy - info_soma['centroid'][:2]
        v1 = df_rois.loc[roi_id1].roi_coords_stack_xy - info_soma['centroid'][:2]
    
    elif dim == 3:

        v0 = df_rois.loc[roi_id0].roi_coords - info_soma['centroid']
        v1 = df_rois.loc[roi_id1].roi_coords - info_soma['centroid']
        
    c =  v0 @ v1 / np.linalg.norm(v0) / np.linalg.norm(v1)
    
    deg = np.degrees((np.arccos(np.clip(c, -1.0, 1))))
    
    return deg


###########################
## RF quality: Moran's I ##
###########################

def MoransI(M):
    
    ndims = M.shape
    M = M.flatten()
    
    import numpy as np

    def adjacency_matrix(ndims):
        
        n, m = ndims  
        
        xes = np.zeros([n, m])
        yes = np.zeros([n, m])
        for x in range(n):
            for y in range(m):
                xes[x, y] = x
                yes[x, y] = y
        xes = xes.flatten()
        yes = yes.flatten()
        
        adjM = np.zeros([n*m,n*m])   
        for i in range(n):
            xdiff = i - xes
            for j in range(m):
                ydiff = j - yes
                diff = np.sqrt(xdiff**2 + ydiff**2)
                diff[diff > 1] =0
                adjM[i*m + j] = diff
                
        return adjM 
        
    
    w = adjacency_matrix(ndims)

    mbar = np.mean(M)

    P1 = 0
    P2 = 0
    P3 = 0
    for i in range(ndims[0]*ndims[1]):
        for j in range(ndims[0]*ndims[1]):
            P1 += (w[i,j] * (M[i] - mbar) * (M[j] - mbar))
            P2 += w[i,j]
        P3 += (M[i] - mbar)**2

    P4 = ndims[0]*ndims[1] / P2
    I = P4 * P1 / P3
    
    return I

##################################
## RF quality: noise esitmation ##
##################################

def estimate_noise(I):
    from scipy.signal import convolve2d

    H, W = I.shape

    M = [[1, -2, 1],
       [-2, 4, -2],
       [1, -2, 1]]

    sigma = np.sum(np.sum(np.absolute(convolve2d(I, M))))
    sigma = sigma * np.sqrt(0.5 * np.pi) / (6 * (W-2) * (H-2))

    return sigma


#########################################
## RF quality: correlation coefficient ##
#########################################

def spectral_corrcoef(I):

    import numpy
    def get_grids(N_X, N_Y):
        from numpy import mgrid
        return mgrid[-1:1:1j*N_X, -1:1:1j*N_Y]

    def frequency_radius(fx, fy):
        R2 = fx**2 + fy**2
        (N_X, N_Y) = fx.shape
        R2[N_X/2, N_Y/2]= numpy.inf

        return numpy.sqrt(R2)

    def enveloppe_color(fx, fy, alpha=1.0):
        # 0.0, 0.5, 1.0, 2.0 are resp. white, pink, red, brown noise
        # (see http://en.wikipedia.org/wiki/1/f_noise )
        # enveloppe
        return 1. / frequency_radius(fx, fy)**alpha #

    import scipy
    N_X, N_Y = I.shape
    fx, fy = get_grids(N_X, N_Y)
    pink_spectrum = enveloppe_color(fx, fy)

    from scipy.fftpack import fft2
    power_spectrum = numpy.abs(fft2(image))**2

    return power_spectrum