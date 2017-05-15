import h5py
import glob
import skfmm
import scipy.ndimage
import scipy.misc

import numpy as np
import tifffile as tiff
import pandas as pd

from skimage.feature import match_template

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

    mask = np.logical_and(ballvolume, bimg)
    
    soma = {'centroid': centroid, 
            'radius': radius, 
            'mask': mask}

    return soma


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
                               'roi_coords',
                               'filepath'))
    
    rec_id = 0
    roi_id = 0
    roi_rel_id = 0

    for file in dendrite_files:
        
        roi_coords_stack_xy = {}

        dname = file.split('/')[-1].split('_')[2]

        d = load_h5_data(file)
        d_rec_rot = rotate_rec(d, stack_data)
        d_rois_rot, roi_coords_rot = rotate_roi(d, stack_data)

        d_rel_cy, d_rel_cx, d_rel_cz = rel_position_um(soma_data, d) / pixel_size_stack(stack_data)
        d_stack_cx, d_stack_cy = int(stack_soma_cx+d_rel_cx), int(stack_soma_cy+d_rel_cy) 

        padding = int(max(d_rec_rot.shape)) 
        crop = linestack_xy[d_stack_cx-padding:d_stack_cx+padding, d_stack_cy-padding:d_stack_cy+padding]

        d_rec_rot_x0, d_rec_rot_y0 = roi_matching(crop, d_rec_rot) # coords of roi in crop
        roi_coords_crop = roi_coords_rot + np.array([d_rec_rot_x0, d_rec_rot_y0])

#         d_rec_rot_origin[dname] = np.array([d_rec_rot_x0 + d_stack_cx-padding , d_rec_rot_y0 + d_stack_cy-padding])

        roi_coords_stack_xy[dname] = roi_coords_crop + np.array([d_stack_cx-padding, d_stack_cy-padding])

        roi_coords_stack_xyz = {}

        keys = roi_coords_stack_xy.keys()
        
        scope = 0

        for key in keys:
            d_coords_xy = np.round(roi_coords_stack_xy[key]).astype(int)
            d_coords_xyz = np.zeros([d_coords_xy.shape[0], d_coords_xy.shape[1]+1])
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
    
                x_o = np.round(roi_coords_stack_xy[dname][roi_rel_id][0]).astype(int)
                y_o = np.round(roi_coords_stack_xy[dname][roi_rel_id][1]).astype(int)
                z_o = np.mean(offset[2]).astype(int)
    

                candidates = np.array([np.array([x_c[i], y_c[i], z_c[i]]) for i in range(len(x_c))])
                origins = np.array([x_o, y_o, z_o])
                
                x, y, z = candidates[np.argmin(np.sum((candidates - origins) ** 2, 1))]
    
                df_rois.loc[roi_id] = [rec_id, roi_id, roi_rel_id, tuple([x, y, z]), file]

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
        p_id.append(roi_on_which_path(df_trace, ROI_coords[i]))
    df_rois['path_id'] = p_id
    
    
    ##############################################################
    ## Get all branch points, segments and distance from each ROI to soma ##
    ##############################################################

    df_paths = get_df_paths(df_trace, soma_info)

    branchpoints = {}
    segments = {}
    distances = {}

    for roi_id in df_rois.roi_id:

        sms, smlen, bpts, num_bpts = get_all_segments(roi_id, df_rois, df_paths)
        
        if len(bpts) >= 1 and bpts != [[]]:
            # print(bpts)
            bpts = np.vstack([p for p in bpts if p != []])
            
        branchpoints[int(roi_id)] = bpts
        segments[int(roi_id)] = sms
        distances[int(roi_id)] = np.sum(smlen) * stack_pixel_size
        
        # print(roi_id, np.sum(smlen))

    df_rois['distance_from_soma_um'] = distances.values() 
    df_rois['branchpoints'] = branchpoints.values()
    df_rois['segments'] = segments.values()

    return df_rois, df_paths

def roi_on_which_path(df_trace, roi_coord, verbose=False):

    '''
    Get the id of a path which a ROI is on.
    
    Paramenters
    ===========
    df_trace: pandas DataFrame
        - DataFrame of the trace.
    roi_coord:
        the xyz-coordinate of a ROI
    
    Return
    ======
    path_id: str
        the id of a path
    '''
    
    path_list = np.unique(df_trace['id'].tolist())
    
    for path_id in path_list:
        
        path_coords = get_path_coords(df_trace, path_id)
        if (roi_coord == path_coords).all(1).any():
            if verbose:
                print('ROI {} is on path {}'.format(roi_coord, path_id))
            return int(path_id)

    print('ROI {} is not on any paths'.format(roi_coord))



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

def connected_with_soma(all_paths, key, soma_info, threshold=10):
    
    path = all_paths[key]
    
    xm, ym, zm = np.ma.where(soma_info['mask'])
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

        if connected_with_soma(all_paths, key, soma_info, threshold=10):

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

def distance_point2end(point, path_id, df_paths):
    
    path = df_paths.loc[path_id].path
    point_loc = np.where((point  == path).all(1))[0]
    
    if len(point_loc) > 1:
        point_loc = point_loc[0]
        
    segment = path[:int(point_loc), :]
    segment_length = np.sum(np.sqrt(np.sum((segment[1:] - segment[:-1])**2, 1)))    
    
    return segment_length, segment

def get_segment(connected_points, point_of_interest, path_id, df_paths):
    
    if len(connected_points) != 0:
        
        points_on_path = np.vstack([point_of_interest, connected_points])
    
        results =[distance_point2end(point, path_id, df_paths) for point in points_on_path]
        distances = [results[i][0] for i in range(len(results))]
        
        segment = results[0][1]
        segment_length = distances[0]
        
        if sum(segment_length > distances) != 0:
            branchpoints = points_on_path[segment_length > distances]
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
    
    # i=0

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

        # i+=1
        
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

##### something for plot
#####
# mask = ~soma_info['mask'].any(2)
# soma_to_mask = soma_info['mask'].mean(2)
# soma_to_mask[soma_to_mask != 0] = 1
# soma_masked = np.ma.masked_array(soma_to_mask, mask)

# plt.figure(figsize=(16,16))
# plt.imshow(linestack.mean(2), origin='lower', cmap=plt.cm.binary)
# plt.grid('off')

# plt.scatter(ROI_coords[:, 1], ROI_coords[:, 0], c='red')

# # target_path_id = 9
# roi_id = 1
# target_path_id = df_rois.loc[roi_id].path_id
# ps = df_rois.loc[roi_id].roi_coords
# plt.scatter(ps[1], ps[0], color='orange')
# sms, smlen, bpts, num_bpts = get_all_segments(roi_id, df_rois, df_paths)
# # print(sl, bp)

# target_path = df_paths[df_paths.path_id == target_path_id].path.values[0]
# plt.plot(target_path[:, 1], target_path[:, 0], color='black')
# plt.annotate(target_path_id, xy=(target_path[-1, 1]-5, target_path[-1, 0]-5), color='black', fontsize=20)

# # connected_path_ids = df_paths[df_paths.path_id == target_path_id].connected_by.values[0]
# # connected_path_ids
# # for connected_path_id in connected_path_ids:
# #     connected_path = df_paths[df_paths.path_id == connected_path_id].path.values[0]
    
# #     connected_point = df_paths[df_paths.path_id == connected_path_id].connect_to_at.values[0]
    
# #     plt.plot(connected_path[:, 1], connected_path[:, 0], color='blue')
# #     plt.scatter(connected_point[1],connected_point[0], color='blue')
# #     plt.annotate(connected_path_id, xy=(connected_path[-1, 1]-5, connected_path[-1, 0]-5), color='black', fontsize=20)

# # from copy import copy
# # palette = copy(plt.cm.gray)
# # palette.set_over('w', 0)
# # palette.set_under('r', 0.5)
# # palette.set_bad('w')

# # plt.imshow(np.ma.make_mask(soma_info['mask'].mean(2)), )
# plt.imshow(soma_masked, origin='lower', cmap=plt.cm.viridis, vmin=0.0)



# for smm in sms:
#     plt.plot(smm[:, 1], smm[:,0])
    
# bpts = np.vstack([p for p in bpts if p != []])
# plt.scatter(bpts[:, 1], bpts[:, 0], color='blue')

def get_soma_xy(stack_data):
    soma_info = get_soma_info(stack_data['wDataCh0'])
    mask = ~soma_info['mask'].any(2)
    soma_to_mask = soma_info['mask'].mean(2)
    soma_to_mask[soma_to_mask != 0] = 1
    soma_masked = np.ma.masked_array(soma_to_mask, mask)
    
    return soma_masked