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

    dendrites_h5_paths = []
    for file in os.listdir(celldir + '/Pre'):
        if ('_d' in file.lower() and 'chirp' not in file.lower() and 's_d' not in file.lower()):
            dendrites_h5_paths.append(celldir + '/Pre/' + file)
            
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


def get_info_soma(stack):
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

    threshold = np.mean(stack) + np.std(stack)
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

    soma_to_be_masked = mask_3d.mean(2)
    soma_to_be_masked[soma_to_be_masked !=0] = 1
    mask_2d = np.ma.masked_array(soma_to_be_masked, ~mask_3d.any(2))
    
    soma = {'centroid': centroid, 
            'radius': radius, 
            'mask_2d': mask_2d,
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

def rotate_rec(rec, stack):
    
    ang_deg = rec['wParamsNum'][31] # ratoate angle (degree)
    ang_rad = ang_deg * np.pi / 180 # ratoate angle (radian)
    
    rec_rec = resize_rec(rec, stack)
    rec_rot = scipy.ndimage.interpolation.rotate(rec_rec, ang_deg)
    
    return rec_rot

def rec_preprop(rec):
    
    reci = rec['wDataCh0'].mean(2)
    reci[:4, :] = reci.mean() - 0.5*reci.std()
    
    return reci

def rotate_roi(rec, stack):
    
    ang_deg = rec['wParamsNum'][31] # ratoate angle (degree)
    ang_rad = ang_deg * np.pi / 180 # ratoate angle (radian)
    
    rec_rois = resize_roi(rec, stack)
    rec_rois_rot = scipy.ndimage.interpolation.rotate(rec_rois, ang_deg)
    
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

def get_connect_to(all_paths, key):
    
    path = all_paths[key]
    
    sub_paths = all_paths.copy()
    sub_paths.pop(key)
    
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

    roi0_on_path_id = point_on_which_path(df_trace, roi0)
    roi1_on_path_id = point_on_which_path(df_trace, roi1)

    bpts0 = df_rois.loc[roi_id0].branchpoints
    bpts1 = df_rois.loc[roi_id1].branchpoints

    overlap_pts = [pt0 for pt0 in bpts0 for pt1 in bpts1  if (pt0 == pt1).all()]
    
    sm_roi0_to_soma = df_rois.loc[roi_id0].segments
    sm_roi1_to_soma = df_rois.loc[roi_id1].segments
    
    if len(overlap_pts)>0: # intersected
        
        nexus_pt = overlap_pts[0]
        
        nexus_loc_in_bpts0 = np.where([(nexus_pt == pt).all() for pt in bpts0])[0][0]
        nexus_loc_in_bpts1 = np.where([(nexus_pt == pt).all() for pt in bpts1])[0][0]

        sm0 = get_segment_from_roi_to_nexus_pt(sm_roi0_to_soma, nexus_pt)
        sm1 = get_segment_from_roi_to_nexus_pt(sm_roi1_to_soma, nexus_pt) 

        if sm0.shape[0] > sm1.shape[0]:
            A = sm0.copy()
            B = sm1.copy()
        else:
            A = sm1.copy()
            B = sm0.copy()

        loc = np.where((B[-1] == A).all(1))[0] # loc of shorter roi on longer roi

        if len(loc) == 0:
            sms_between = [sm0, sm1]
            bpts_between = np.vstack([bpts0[:nexus_loc_in_bpts0+1], bpts1[:nexus_loc_in_bpts1+1]])
        else:
            sms_between = [A[loc[0]:]]
            bpts_between = np.vstack([bpts0[:nexus_loc_in_bpts0], bpts1[:nexus_loc_in_bpts1]])
    
    else: # not intersected

        sm0 = np.vstack([info_soma['centroid'], sm_roi0_to_soma])
        sm1 = np.vstack([info_soma['centroid'], sm_roi1_to_soma])
        sms_between = [sm0, sm1]
        bpts_between = np.vstack([bpts0, info_soma['centroid']]) 
        bpts_between = np.vstack([bpts_between, bpts1])

    bpts_between = unique_row(bpts_between)

    return sms_between, bpts_between