import h5py
import glob
import skfmm
import scipy.ndimage
import scipy.misc

import numpy as np
import tifffile as tiff
import pandas as pd

from skimage.feature import match_template

import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib import rcParams
rcParams['font.family'] = 'DejaVu Sans'

#######################
## Helper functions ##
######################

def load_h5_data(file_name):
    with h5py.File(file_name,'r') as f:
        return {key:f[key][:] for key in list(f.keys())}


####################################
## Get trace-related info from trace file ##
####################################

import pandas as pd
import numpy as np

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


def get_soma_info(stack):
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
        In [1]: soma = detect_soma(stack['wDataCh0'])
        
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

# def get_soma_xy(stack_data):
#     soma_info = get_soma_info(stack_data['wDataCh0'])
    
#     mask = ~soma_info['mask'].any(2)
#     soma_to_mask = soma_info['mask'].mean(2)
#     soma_to_mask[soma_to_mask != 0] = 1
#     soma_masked = np.ma.masked_array(soma_to_mask, mask)
    
#     return soma_masked


####################################################
## Functions for finding ROI coordinates on Stack ##
####################################################

def pixel_size_rec(rec, verbose=False):
    
    """
    Return the real length (in um) of each pixel point.
    """
    len_rec_x_pixel = 64
    len_rec_x_um = 71.5 / rec['wParamsNum'][30]
    
    rec_pixel_size = len_rec_x_um / len_rec_x_pixel
    
    if verbose:
        print("the real length of each pixel in this recording is: \n{0} um".format(rec_pixel_size))
    
    return rec_pixel_size


def pixel_size_stack(stack, verbose=False):
    
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
    
    rec_pixel_size = pixel_size_rec(rec)
    stack_pixel_size = pixel_size_stack(stack)
    
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
    
    labels = np.unique(rec_rois)[:-1]

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

def get_df_rois(trace_path, dendrite_files, soma_h5_path, stack_h5_path):
    
    df_trace, meta_trace = read_trace(trace_path)
    
    soma_data = load_h5_data(soma_h5_path)
    stack_data = load_h5_data(stack_h5_path)
    
    stack_pixel_size = pixel_size_stack(stack_data)

    soma_info = get_soma_info(stack_data['wDataCh0']) # pixel
    
    (stack_soma_cx, stack_soma_cy, stack_soma_cz) = soma_info['centroid']
    linestack = trace2linestack(df_trace, meta_trace)
    linestack_xy = linestack.mean(2)
    
    df_rois = pd.DataFrame(columns=('recording_id', 
                               'roi_id', 
                               'roi_rel_id',
                               'recording_center', 
                               'roi_coords',

                               'STRF_SVD_Space0',
                               'STRF_SVD_Time0',
                               'Traces0_raw',
                               'Traces0_znorm',
                               'Tracetimes0',
                               'Triggertimes',
                               'Triggervalues',
                               # 'filepath',
                               ))
    
    rec_id = 0
    roi_id = 0
    roi_rel_id = 0

    for file in dendrite_files:

        dname = file.split('/')[-1].split('_')[2]

        d = load_h5_data(file)
        d_rec_rot = rotate_rec(d, stack_data)
        d_rois_rot, roi_coords_rot = rotate_roi(d, stack_data)

        d_rel_cy, d_rel_cx, d_rel_cz = rel_position_um(soma_data, d) / pixel_size_stack(stack_data)
        d_stack_cx, d_stack_cy = int(stack_soma_cx+d_rel_cx), int(stack_soma_cy+d_rel_cy) 

        padding = int(max(d_rec_rot.shape)) 
        crop = linestack_xy[d_stack_cx-padding:d_stack_cx+padding, d_stack_cy-padding:d_stack_cy+padding]

        scale_down = 0.9
        while 0 in np.unique(crop.shape):
            padding = int(scale_down * max(d_rec_rot.shape))
            crop = linestack[d_stack_cx-padding:d_stack_cx+padding, d_stack_cy-padding:d_stack_cy+padding].mean(2)
            scale_down *= scale_down

        d_rec_rot_x0, d_rec_rot_y0 = roi_matching(crop, d_rec_rot) # coords of roi in crop
        roi_coords_crop = roi_coords_rot + np.array([d_rec_rot_x0, d_rec_rot_y0])
        rec_center_crop = np.array([d_rec_rot.shape[0]/2,  d_rec_rot.shape[1]/2]) + np.array([d_rec_rot_x0, d_rec_rot_y0])


        roi_coords_stack_xy = roi_coords_crop + np.array([d_stack_cx-padding, d_stack_cy-padding])
        rec_center_stack_xy = rec_center_crop + np.array([d_stack_cx-padding, d_stack_cy-padding])
        
        scope = 0

        d_coords_xy = np.round(roi_coords_stack_xy).astype(int)
        d_coords_xyz = np.zeros([d_coords_xy.shape[0], d_coords_xy.shape[1]+1])
        
        rois_labels_real = np.arange(0, len(d_coords_xy))[::-1]

        for i, roi_xy in enumerate(d_coords_xy):
            
            offset = np.where(linestack[roi_xy[0]-scope:roi_xy[0]+scope+1, roi_xy[1]-scope:roi_xy[1]+scope+1] == 1)

            if offset[2].size == 0:
                scope += 1
                offset = np.where(linestack[roi_xy[0]-scope:roi_xy[0]+scope+1, roi_xy[1]-scope:roi_xy[1]+scope+1] == 1)
                if offset[2].size == 0:
                    continue
                else:
                    x_c = np.arange(roi_xy[0]-scope,roi_xy[0]+scope+1)[offset[0]]
                    y_c = np.arange(roi_xy[1]-scope,roi_xy[1]+scope+1)[offset[1]]
                    z_c = offset[2]
            else:
                x_c = np.arange(roi_xy[0]-scope,roi_xy[0]+scope+1)[offset[0]]
                y_c = np.arange(roi_xy[1]-scope,roi_xy[1]+scope+1)[offset[1]]
                z_c = offset[2]

            x_o = np.round(roi_coords_stack_xy[roi_rel_id][0]).astype(int)
            y_o = np.round(roi_coords_stack_xy[roi_rel_id][1]).astype(int)
            z_o = np.mean(offset[2]).astype(int)


            candidates = np.array([np.array([x_c[i], y_c[i], z_c[i]]) for i in range(len(x_c))])
            origins = np.array([x_o, y_o, z_o])
            
            x, y, z = candidates[np.argmin(np.sum((candidates - origins) ** 2, 1))]

            roi_label = rois_labels_real[i]

            STRF_SVD_Space0 = d['STRF_SVD_Space0'][:, :, roi_label]
            STRF_SVD_Time0 = d['STRF_SVD_Time0'][:, roi_label]

            Traces0_raw = d['Traces0_raw'][:, roi_label]
            Traces0_znorm = d['Traces0_znorm'][:, roi_label]
            Tracetimes0 = d['Tracetimes0'][:, roi_label]
            Triggertimes = d['Triggertimes']
            Triggervalues = d['Triggervalues']

            # df_rois.loc[roi_id] = [rec_id, roi_id, roi_rel_id, tuple([x, y, z]), file]
            df_rois.loc[roi_id] = [rec_id, roi_id, roi_rel_id, rec_center_stack_xy, np.array([x, y, z]), STRF_SVD_Space0, STRF_SVD_Time0, Traces0_raw, Traces0_znorm, Tracetimes0, Triggertimes, Triggervalues]

            roi_id += 1
            roi_rel_id += 1

        roi_rel_id= 0
        rec_id += 1


    ########################################
    ## Find the path which each ROI is on ##
    ########################################


    ROI_coords = np.array(df_rois.roi_coords.tolist())
    p_id = []
    for i in range(len(ROI_coords)):
        # p_id.append(roi_on_which_path(df_trace, ROI_coords[i]))
        p_id.append(point_on_which_path(df_trace, ROI_coords[i]))
    df_rois['path_id'] = p_id
    
    
    ##############################################################
    ## Get all branch points, segments and distance from each ROI to soma ##
    ##############################################################

    # df_paths = get_df_paths(df_trace, soma_info)
    df_paths = get_df_paths2(df_trace, soma_info)

    branchpoints = {}
    segments = {}
    distance_dendritic = {}
    distance_radial = {}
    branchpoints_distances = {}

    for roi_id in df_rois.roi_id:

        sms, smlen, bpts, num_bpts = get_all_segments(roi_id, df_rois, df_paths)
        
        if len(bpts) >= 1 and bpts != [[]]:
            # print(bpts)
            bpts = np.vstack([p for p in bpts if p != []])

        branchpoints[int(roi_id)] = bpts        
        segments[int(roi_id)] = sms
        distance_dendritic[int(roi_id)] = np.sum(smlen) * stack_pixel_size
        distance_radial[int(roi_id)] = np.sqrt(np.sum((soma_info['centroid'] - df_rois.loc[roi_id].roi_coords) ** 2)) * stack_pixel_size
        
        # print(roi_id, np.sum(smlen))

    df_rois['distance_dendritic_um'] = distance_dendritic.values() 
    df_rois['distance_radial_um'] = distance_radial.values()
    df_rois['branchpoints'] = branchpoints.values()
    df_rois['segments'] = segments.values()

    ## turn ids into int
    df_rois['recording_id'] = df_rois['recording_id'].astype(int)
    df_rois['roi_id'] = df_rois['roi_id'].astype(int)
    df_rois['roi_rel_id'] = df_rois['roi_rel_id'].astype(int)

    df_branchpoints = get_df_branchpoints(df_trace, df_rois, df_paths, stack_pixel_size)

    return df_rois, df_paths, df_branchpoints

def get_df_branchpoints(df_trace, df_rois, df_paths, stack_pixel_size):
    
    branchpoints_tmp = df_rois.branchpoints.values
    branchpoints_tmp = np.array([bpt for bpt in branchpoints_tmp if bpt != np.array([])])

    tmp = np.vstack(branchpoints_tmp)
    
    all_branchpoints = tmp[np.unique(tmp.view(np.void(tmp.strides[0])),1)[1]]
    
    df_branchpoints = pd.DataFrame(columns=(
                                            'branchpoint_id',
                                            'branchpoint',
                                            'branchpoint_to_soma_um'))
    for i, bpt in enumerate(all_branchpoints):
        
        path_id = point_on_which_path(df_trace, bpt)
        
        results = distance_point2end(bpt, path_id, df_paths)
        segment_lengths = results[0]
        
        while df_paths.loc[path_id].connect_to != -1:
            
            point_of_interest = df_paths.loc[path_id].connect_to_at
            path_id = df_paths.loc[path_id].connect_to
            
            results = distance_point2end(point_of_interest, path_id, df_paths)
            segment_length = results[0]
            segment_lengths += segment_length
            
        df_branchpoints.loc[i] = [i, bpt, segment_lengths * stack_pixel_size] 
        
        df_branchpoints['branchpoint_id'] = df_branchpoints['branchpoint_id'].astype(int)
                
    return df_branchpoints

# def roi_on_which_path(df_trace, roi_coord, verbose=False):

#     '''
#     Get the id of a path which a ROI is on.
    
#     Paramenters
#     ===========
#     df_trace: pandas DataFrame
#         - DataFrame of the trace.
#     roi_coord:
#         the xyz-coordinate of a ROI
    
#     Return
#     ======
#     path_id: str
#         the id of a path
#     '''
    
#     path_list = np.unique(df_trace['id'].tolist())
    
#     for path_id in path_list:
        
#         path_coords = get_path_coords(df_trace, path_id)
#         if (roi_coord == path_coords).all(1).any():
#             if verbose:
#                 print('ROI {} is on path {}'.format(roi_coord, path_id))
#             return int(path_id)

#     print('ROI {} is not on any paths'.format(roi_coord))

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


####################################
## Functions for Trace Statistics ##
####################################



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

def get_df_paths(df_trace, soma_info):
    
    all_paths = get_all_paths(df_trace)
    df_paths = pd.DataFrame(list(all_paths.items()), columns=['path_id', 'path'])

    connect_to_all    = []
    connect_to_at_all = []

    for i, key in enumerate(all_paths.keys()):

        if connected_with_soma(all_paths, key, soma_info, threshold=5):

            connect_to    = -1
            connect_to_at = []
        else:
            connect_to, connect_to_at = get_connect_to(all_paths, key)

        connect_to_all.append(connect_to)
        connect_to_at_all.append(connect_to_at)
    
    df_paths['connect_to'] = connect_to_all
    df_paths['connect_to_at'] = connect_to_at_all
    
    df_paths.sort_values(['path_id'], ascending=[True], inplace=True)
    df_paths.index = df_paths.path_id.as_matrix()

    connected_by_all    = []
    connected_by_at_all = []
    
    for i, key in enumerate(df_paths.path_id):
        
        connected_by    = df_paths[df_paths.connect_to == key].path_id.tolist()
        connected_by_at = df_paths[df_paths.connect_to == key].connect_to_at.tolist()
    
        connected_by_all.append(connected_by)
        connected_by_at_all.append(connected_by_at)
    
    df_paths['connected_by'] = connected_by_all
    df_paths['connected_by_at'] = connected_by_at_all
    
    return df_paths

def get_df_paths2(df_trace, soma_info):
    
    all_paths = get_all_paths(df_trace)
    df_paths = pd.DataFrame(list(all_paths.items()), columns=['path_id', 'path'])
    # df_paths['path2'] = ''

    connect_to_all    = []
    connect_to_at_all = []

    for i, key in enumerate(all_paths.keys()):

        if connected_with_soma(all_paths, key, soma_info, threshold=5):

            connect_to    = -1
            connect_to_at = []
        else:
            connect_to, connect_to_at = get_connect_to(all_paths, key)

        connect_to_all.append(connect_to)
        connect_to_at_all.append(connect_to_at)
    
    df_paths['connect_to'] = connect_to_all
    df_paths['connect_to_at'] = connect_to_at_all
    
    df_paths.sort_values(['path_id'], ascending=[True], inplace=True)
    df_paths.index = df_paths.path_id.as_matrix()

    connected_by_all    = []
    connected_by_at_all = []
    
    for i, key in enumerate(df_paths.path_id):
        
        connected_by    = df_paths[df_paths.connect_to == key].path_id.tolist()
        connected_by_at = df_paths[df_paths.connect_to == key].connect_to_at.tolist()
    
        connected_by_all.append(connected_by)
        connected_by_at_all.append(connected_by_at)
    
    df_paths['connected_by'] = connected_by_all
    df_paths['connected_by_at'] = connected_by_at_all
    
    path2 = {}
    for path_id in df_paths.path_id:
    
        connect_to = df_paths.loc[path_id].connect_to
        path = df_paths.loc[path_id].path
        connect_to_at = df_paths.loc[path_id].connect_to_at
        
        if connect_to == -1:
            path2[path_id] = path

        else:
            if (connect_to_at == path).all(1).any():
                path2[path_id] = path
            else:
                path = np.vstack([connect_to_at, path])
                path2[path_id] = path
    
    df_paths['path'] = path2.values()

    return df_paths

def distance_point2end(point, path_id, df_paths):
    
    path = df_paths.loc[path_id].path
    point_loc = np.where((point  == path).all(1))[0]
    
    if len(point_loc) > 1:
        point_loc = point_loc[0]
        
    # segment = path[:int(point_loc), :] 
    segment = path[:int(point_loc), :] # should include the point_loc into the segment?
    segment_length = np.sum(np.sqrt(np.sum((segment[1:] - segment[:-1])**2, 1)))    
    
    return segment_length, segment

def get_segment(connected_points, point_of_interest, path_id, df_paths):
    
    if len(connected_points) != 0:
        
        points_on_path = np.vstack([point_of_interest, connected_points])
    
        results =[distance_point2end(point, path_id, df_paths) for point in points_on_path]
        # results =[(segment_length0, segment0), (segment_length1, segment1), ...]
        distances = np.array([results[i][0] for i in range(len(results))]) # lengths of all points to the end of the path

        segment = results[0][1] # the segment from poi to the end of the path 
        segment_length = distances[0] # the length from poi to the end of the path

        # new added (need more time to review)
        change_order = np.argsort(distances)[::-1]
        # print(distances)

        distances_sorted = distances[change_order]
        points_on_path_sorted = points_on_path[change_order]
        
        if sum(segment_length > distances_sorted) != 0:
            branchpoints = points_on_path_sorted[segment_length > distances_sorted]        

        else:
            branchpoints = []
        num_branchpoints = len(branchpoints)
    else:
        
        point_on_path = point_of_interest
        results = distance_point2end(point_on_path, path_id, df_paths)
        segment_length = results[0]
        segment  = results[1]
        branchpoints = []
#         num_branchpoints = 0
        num_branchpoints = len(branchpoints)
        
    return segment, segment_length, branchpoints, num_branchpoints

def get_all_segments(roi_id, df_rois, df_paths):

    path_id = df_rois.loc[roi_id].path_id
    connected_points = df_paths.loc[path_id].connected_by_at
    roi_coord = df_rois.loc[roi_id].roi_coords
    
    segment, segment_length, branchpoints, num_branchpoints = get_segment(connected_points, roi_coord, path_id, df_paths)
    
    all_segments = [segment]
    all_segments_length = [segment_length]
    all_branchpoints = [branchpoints]
    sum_branchpoints = num_branchpoints

    while df_paths.loc[path_id].connect_to != -1:
        
        point_of_interest = df_paths.loc[path_id].connect_to_at
        path_id = df_paths.loc[path_id].connect_to
        
        connected_points = df_paths.loc[path_id].connected_by_at
        
        segment, segment_length, branchpoints, num_branchpoints = get_segment(connected_points, point_of_interest, path_id, df_paths)
        
        if len(branchpoints) != 0:
            branchpoints = np.vstack([point_of_interest, branchpoints])
        else: 
            branchpoints = point_of_interest
        
        num_branchpoints += 1
        
        all_segments.append(segment)    
        all_segments_length.append(segment_length)
        all_branchpoints.append(branchpoints)
        sum_branchpoints += num_branchpoints
        
    return all_segments, all_segments_length, all_branchpoints, sum_branchpoints

def get_distance_from_roi_to_soma(df_rois, roi_id, unit='pixel'):
    
    segments = df_rois.loc[roi_id].segments
    
    distance_arr = []
    for segment in segments:
        distance = np.sum(np.sqrt(np.sum((segment[1:] - segment[:-1])**2, 1)))
        distance_arr.append(distance)
    
    if unit=='pixel':
        return np.sum(distance_arr)
    elif unit=='um':
        return np.sum(distance_arr) * 0.665

def get_distance_from_branchpoint_to_soma(df_rois, roi_id):
    
    bpts = df_rois.loc[roi_id].branchpoints
    sms = df_rois.loc[roi_id].segments
    
    num_bpts = len(bpts)
    num_sms = len(sms)
    
    results = {}
    for i, bpt in enumerate(bpts):
        for j, sm in enumerate(sms):
#             print(i, j, bpt)
            point_loc = np.where((bpt == sm).all(1))[0]
            
            if len(point_loc) == 0:
                continue
            else:
#                 print(i, j, bpt, point_loc, type(point_loc))
                bpt_segment = sm[:int(point_loc), :]
                bpt_segment_length = np.sum(np.sqrt(np.sum((bpt_segment[1:] - bpt_segment[:-1])**2, 1)))
        
                if j+1 == num_sms:
                    distance_from_bpt_to_soma = bpt_segment_length
                    results[i] = distance_from_bpt_to_soma
                    break
                    
                else:
                    sms_rest = sms[j:]
                    
                    sms_len_rest = 0
                    for sm_rest in sms_rest:
                        sm_len_rest = np.sum(np.sqrt(np.sum((sm_rest[1:] - sm_rest[:-1])**2, 1)))
                        sms_len_rest += sm_len_rest
                        
                    distance_from_bpt_to_soma = bpt_segment_length + sms_len_rest
                    results[i] = distance_from_bpt_to_soma
                    break
    
    return results

#########################
## Pairwise Statistics ##
#########################

def unique_row(a):
    
    b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
    _, idx = np.unique(b, return_index=True)
    
    unique_a = a[idx]
    
    return unique_a

def get_segment_from_roi_to_nexus_pt(sms_roi_to_soma, nexus_pt):    
    
    all_locs = [(pt == nexus_pt).all(1) for pt in sms_roi_to_soma]    
    for i, loc in enumerate(all_locs):
        if loc.any():
            nexus_pt_loc = np.where(loc)[0][0]
            sm_roi_to_nexus = sms_roi_to_soma[i][nexus_pt_loc:]
            break
                
    if i == 0:
        sm_f = sm_roi_to_nexus
        sm_f = sm_f.reshape(1, sm_f.shape[0], sm_f.shape[1])
#         print(sm_f.shape)
        
        return np.vstack(sm_f)
    else:
        
        sm0 = sms_roi_to_soma[:i]
        sm1 = sm_roi_to_nexus
        
        sm_f = sm0.copy()
        sm_f.append(sm1)
        sm_f = np.array(sm_f)
                
        return np.vstack(sm_f[::-1])

def get_info_roi2roi(df_trace, df_paths, df_rois, df_branchpoints, roi_id0, roi_id1):
    
    roi0 = df_rois.loc[roi_id0].roi_coords
    roi1 = df_rois.loc[roi_id1].roi_coords

    roi0_on_path_id = point_on_which_path(df_trace, roi0)
    roi1_on_path_id = point_on_which_path(df_trace, roi1)

    bpts0 = df_rois.loc[roi_id0].branchpoints
    bpts1 = df_rois.loc[roi_id1].branchpoints

    overlap_pts = [pt0 for pt0 in bpts0 for pt1 in bpts1  if (pt0 == pt1).all()]
    
    sms_roi0_to_soma = df_rois.loc[roi_id0].segments
    sms_roi1_to_soma = df_rois.loc[roi_id1].segments
    
    if len(overlap_pts)>0: # intersected
        
        nexus_pt = overlap_pts[0]
        
        nexus_loc_in_bpts0 = np.where([(nexus_pt == pt).all() for pt in bpts0])[0][0]
        nexus_loc_in_bpts1 = np.where([(nexus_pt == pt).all() for pt in bpts1])[0][0]

        sm0 = get_segment_from_roi_to_nexus_pt(sms_roi0_to_soma, nexus_pt)
        sm1 = get_segment_from_roi_to_nexus_pt(sms_roi1_to_soma, nexus_pt) 

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
        
        sm0 = np.vstack(sms_roi0_to_soma[::-1])
        sm1 = np.vstack(sms_roi1_to_soma[::-1])
        sms_between = [sm0, sm1]
        bpts_between = np.vstack([bpts0, bpts1])
        
    bpts_between = unique_row(bpts_between)

    return sms_between, bpts_between

def get_df_pairwise(df_trace, df_paths, df_rois, df_branchpoints):

    df_pairwise = pd.DataFrame(columns=('pair_id',
                                        'segments_between',
                                        'branchpoints_between'))

    all_roi_id = df_rois.roi_id.values

    i = 0
    for roi_id0 in all_roi_id:
        
        all_roi_id = np.delete(all_roi_id, 0)
        
        for roi_id1 in all_roi_id:
            
            sms_between, bpts_between = get_info_roi2roi(df_trace, df_paths, df_rois, df_branchpoints, roi_id0, roi_id1)
            df_pairwise.loc[i] = [set([roi_id0, roi_id1]), sms_between, bpts_between]
            i += 1

    return df_pairwise

#############
## Ploting ##
#############

def plot_all_rois(df_trace, df_rois, meta_trace, soma_info, figsize=(16,16), savefig=False, fig_path='.' ):

    linestack = trace2linestack(df_trace, meta_trace)

    plt.figure(figsize=figsize)
    plt.imshow(linestack.sum(2), origin='lower', cmap=plt.cm.binary)
    plt.imshow(soma_info['mask_2d'], origin='lower', cmap=plt.cm.binary, vmin=0.0, alpha=0.3)

    rois_pos = np.vstack(df_rois.roi_coords)
    rois_dis = df_rois.distance_dendritic_um.values
    plt.scatter(rois_pos[:, 1], rois_pos[:, 0], c=rois_dis/df_rois.distance_dendritic_um.max(), s=180*figsize[0]/16, cmap=plt.cm.viridis)
   
    # plt.colorbar(orientation="horizontal")
    
    for i, roi_id in enumerate(df_rois.roi_id):
        plt.annotate(int(roi_id), xy=(rois_pos[i][1]-3, rois_pos[i][0]-3), color='white', fontsize=figsize[0]*0.7)

    plt.title('ROIs (color-coded by the distance along dendrites to soma)', fontsize=figsize[0])
    plt.grid('off')

    if savefig:
        plt.savefig('{}/ROI_color-coded.png'.format(fig_path))

def plot_anatomy(df_trace,meta_trace, soma_info, figsize=(16,16), savefig=False, fig_path='.' ):

    linestack = trace2linestack(df_trace, meta_trace)

    plt.figure(figsize=figsize)
    plt.imshow(linestack.sum(2), origin='lower', cmap=plt.cm.binary)
    plt.imshow(soma_info['mask_2d'], origin='lower', cmap=plt.cm.binary, vmin=0.0, alpha=0.3)

    plt.grid('off')

def plot_roi2soma(df_trace, df_rois, roi_id, meta_trace, soma_info, figsize=(16,16), savefig=False, fig_path='.' ):

    linestack = trace2linestack(df_trace, meta_trace)

    plt.figure(figsize=figsize)
    plt.imshow(linestack.sum(2), origin='lower', cmap=plt.cm.binary)
    plt.imshow(soma_info['mask_2d'], origin='lower', cmap=plt.cm.binary, vmin=0.0, alpha=0.3)

    all_roi_pos = np.vstack(df_rois.roi_coords)
    plt.scatter(all_roi_pos[:, 1], all_roi_pos[:, 0], color='gray', s=180, alpha=0.3)

    for sm in df_rois.loc[roi_id].segments:

        plt.plot(sm[:, 1], sm[:, 0], color='red', lw=2.5)

    bpts = df_rois.loc[roi_id].branchpoints
    roi_pos = df_rois.loc[roi_id].roi_coords
    plt.scatter(bpts[:, 1], bpts[:, 0], color='black')
    plt.scatter(roi_pos[1], roi_pos[0], color='red', s=180)
    plt.annotate(int(roi_id), xy=(roi_pos[1]-3, roi_pos[0]-3), color='white')

    plt.title('ROI [{}] to soma'.format(roi_id))
    plt.grid('off')

    if savefig:
        plt.savefig('{}/ROI_{}_to_soma.png'.format(fig_path, roi_id))

def plot_roi2roi(df_trace, df_rois, df_pairwise, roi_id0, roi_id1, meta_trace, soma_info, figsize=(16,16), savefig=False, fig_path='.' ):

    linestack = trace2linestack(df_trace, meta_trace)

    plt.figure(figsize=figsize)
    plt.imshow(linestack.sum(2), origin='lower', cmap=plt.cm.binary)
    plt.imshow(soma_info['mask_2d'], origin='lower', cmap=plt.cm.binary, vmin=0.0, alpha=0.3)

    all_roi_pos = np.vstack(df_rois.roi_coords)
    plt.scatter(all_roi_pos[:, 1], all_roi_pos[:, 0], color='gray', s=180, alpha=0.3)

    bpts = df_pairwise[df_pairwise.pair_id == {roi_id0,roi_id1}].branchpoints_between.values[0]
    sms =  df_pairwise[df_pairwise.pair_id == {roi_id0,roi_id1}].segments_between.values[0]


    roi0_pos = df_rois.loc[roi_id0].roi_coords
    roi1_pos = df_rois.loc[roi_id1].roi_coords

    plt.scatter(roi0_pos[1], roi0_pos[0], s=180, color='red')
    plt.scatter(roi1_pos[1], roi1_pos[0], s=180, color='red')

    for bpt in bpts:
        if len(bpt) != 0:
            plt.scatter(bpt[1], bpt[0], s=80, color='black', zorder=10)

    for sm in sms:
        plt.plot(sm[:, 1], sm[:, 0], color='red', lw=4)
        
    if savefig:
        plt.savefig('{}/ROI_{}_to_{}.png'.format(fig_path, roi_id0, roi_id1))


#####################
## Receptive Field ##
#####################

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

def extract_sta_soma(soma_data, stimulus_path):
    
    triggers = soma_data['Triggertimes']
    triggers += -0.1
    weights = interpolate_weights(soma_data, triggers)
    lagged_weights = lag_weights(weights, 5)

    stimulus = load_h5_data(stimulus_path)['k']
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

def resize_RF(RF, stack):
    scale_factor = 30/pixel_size_stack(stack)
    return sp.misc.imresize(RF, size=scale_factor, interp='bicubic')

##########################
## Contours based on RF ##
##########################

def get_roi_contour_on_RF(RF, threshold=3):
    x, y = np.mgrid[:RF.shape[0], :RF.shape[1]]
    c = cntr.Cntr(x,y, RF)

    res = c.trace(RF.mean()+RF.std() * threshold)

    return (res[0][:, 1], res[0][:, 0])

def get_roi_contour_on_stack(df_rois, roi_id, stimulus_path, threshold=1):
    
    rec_id = df_rois.loc[roi_id].recording_id
    
    df_rec = df_rois[df_rois.recording_id == rec_id] 
    
    rec_center = df_rec.recording_center.iloc[0]
    
    RF = extract_sta_rois(df_rec, roi_id, stimulus_path)
#     RF = df_rec.loc[roi_id].STRF_SVD_Space0
    RF = np.fliplr(RF)
    RF_sm = ndimage.gaussian_filter(RF, sigma=(1,1), order=0)
    RF_resized = resize_RF(RF_sm, stack_data)
    
    Rx, Ry = np.array([RF_resized.shape[0]/2, RF_resized.shape[1]/2])
    Sx, Sy = rec_center
    
    left_pad = np.round(Ry - Sy).astype(int)
    bottom_pad = np.round(Rx - Sx).astype(int)
    
    (x, y) = get_roi_contour_on_RF(RF_resized, threshold)
    
    return (x-left_pad, y-bottom_pad)

def get_soma_contour_on_stack(s, soma_info, threshold=2):
    (stack_soma_cx, stack_soma_cy, stack_soma_cz) = soma_info['centroid']
    linestack_xy = soma_info['mask_2d']

    s_rec_rot = tp.rotate_rec(s, stack_data)
    s_rois_rot, roi_coords_rot = tp.rotate_roi(s, stack_data)

    s_rel_cy, s_rel_cx, s_rel_cz = tp.rel_position_um(soma_data, s) / tp.pixel_size_stack(stack_data)
    s_stack_cx, s_stack_cy = int(stack_soma_cx+s_rel_cx), int(stack_soma_cy+s_rel_cy) 

    padding = int(max(s_rec_rot.shape)) 
    crop = linestack_xy[s_stack_cx-padding:s_stack_cx+padding, s_stack_cy-padding:s_stack_cy+padding]

    scale_down = 0.9
    while 0 in np.unique(crop.shape):
        padding = int(scale_down * max(d_rec_rot.shape))
        crop = linestack[d_stack_cx-padding:d_stack_cx+padding, d_stack_cy-padding:d_stack_cy+padding].mean(2)
        scale_down *= scale_down

    s_rec_rot_x0, s_rec_rot_y0 = tp.roi_matching(crop, s_rec_rot) # coords of roi in crop
    rec_center_crop = np.array([s_rec_rot.shape[0]/2,  s_rec_rot.shape[1]/2]) + np.array([s_rec_rot_x0, s_rec_rot_y0])


    rec_center_stack_xy = rec_center_crop + np.array([s_stack_cx-padding, s_stack_cy-padding])

    RF = s['STRF_SVD_Space0'].reshape(15,20)
    RF = np.fliplr(RF)
    RF_sm = ndimage.gaussian_filter(RF, sigma=(1,1), order=0)
    RF_resized = resize_RF(RF_sm, stack_data)
    
    Rx, Ry = np.array([RF_resized.shape[0]/2, RF_resized.shape[1]/2])
    Sx, Sy = rec_center_stack_xy
    
    left_pad = np.round(Ry - Sy).astype(int)
    bottom_pad = np.round(Rx - Sx).astype(int)
    
    (x, y) = get_contour_on_RF(RF_resized, threshold)
    
    return (x-left_pad, y-bottom_pad)