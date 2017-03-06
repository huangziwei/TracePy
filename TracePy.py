import numpy as np
import tifffile as tiff
import pandas as pd

def read_trace(trace_path):
    
    '''
    Read `.trace` file into pandas DataFrame.
    
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
    meta = etree.find('imagesize').attrib
    
    return df, meta

def detect_soma(img_path):
    
    from rivuletpy.soma import Soma
    
    stack = tiff.imread(img_path)
    threshold = np.mean(stack) + np.std(stack)
    bimg = (stack > threshold).astype('int')
    
    soma = Soma()
    soma.detect(bimg, simple=True)
    
    return soma

def get_coords(df, path_id):
    
    x = df[df['id'] == str(path_id)]['x'].astype('int').tolist()
    y = df[df['id'] == str(path_id)]['y'].astype('int').tolist()
    z = df[df['id'] == str(path_id)]['z'].astype('int').tolist()
    coords = np.vstack([[x,y], z]).T
    
    return coords

def get_branches_from_soma(df, soma, theta):
    
    '''
    :theta: scale factor of the soma radius
    '''

    path_dict = {}

    ids = np.unique(df['id'].tolist())

    for idx in ids:

        path_dict[int(idx)] = get_coords(df, idx)
    
    branches_from_soma = []
    
    zm, xm, ym = np.where(soma.mask)
    soma_coords = np.array([xm, ym, zm]).T
    
    for key in path_dict.keys():

        dis = np.sum((path_dict[key][0] - soma_coords) ** 2, 1)
        if (dis < theta * soma.radius * 512 / 340.48).any():
            branches_from_soma.append(key)
            
    return branches_from_soma

def get_closest_branches_from_branch(df, branch_id, id_list, theta):
    

    path_dict = {}
    path_coords = get_coords(df, branch_id)
    
    for idx in id_list:
        path_dict[int(idx)] = get_coords(df, idx)
    
    closest_branches_dict = []
    for key in path_dict.keys():
        dis = np.sum((path_dict[key][0] - path_coords) ** 2, 1)
        if (dis < theta).any():
            closest_branches_dict.append(key)
            
    return closest_branches_dict, path_coords

def drawSphere(xCenter, yCenter, zCenter, r):
    #draw sphere
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x=np.cos(u)*np.sin(v)
    y=np.sin(u)*np.sin(v)
    z=np.cos(v)
    # shift and scale sphere
    x = r*x + xCenter
    y = r*y + yCenter
    z = r*z + zCenter
    return (x,y,z)

def get_all_paths(df):
    
    path_dict = {}
    path_list = np.unique(df['id'].tolist())
    
    for path_id in path_list:
        path_dict[int(path_id)] = get_coords(df, path_id)
    
    return path_dict


def get_distance(a, b):
    
    """
    Euclidean distance between two points.
    """
    
    return np.sqrt(np.sum((a - b ) ** 2, 1))

def connected_with_soma(point, soma_coords, threshold):
    
    """
    Check if a point is connected with soma. 
    """
    
    if np.min(get_distance(point, soma_coords)) < threshold:
        return True
    else:
        return False


def get_pair_distance_from_one_point_to_multiple_paths(point, all_paths):
    
    """
    The Euclidean distance between the point of interest to all paths.
    """
    
    dis_arr = []
    
    for path_id in all_paths.keys():
        dis_shortest = np.min( get_distance( point , all_paths[path_id]))
        dis_arr.append([path_id, dis_shortest])
    
    pair_distance = np.array(dis_arr) 
    
    return pair_distance

def get_the_closest_point(point, target_path_id, all_paths):

    target_path = all_paths[target_path_id]
    pair_distances = get_distance(point, target_path)
    
    return target_path[np.argmin(pair_distances)]
    

def get_the_closest_path(point, all_paths):
    
    pair_distances = get_pair_distance_from_one_point_to_multiple_paths(point, all_paths)
    path_id = pair_distances[np.argmin(pair_distances[:, 1])][0].astype(int)
    
    poc = all_paths[path_id][np.argmin(get_distance(point, all_paths[path_id]))]
    
    return path_id, poc

def get_pair_distance_from_multiple_points_to_one_paths(target_path, all_paths):
    
    """
    The Euclidean distance between the first points of all paths to a target path.
    """
    
    dis_arr = []
    for path_id in all_paths.keys():
        dis_shortest = np.min( get_distance( all_paths[path_id][0] , target_path))
        dis_arr.append([path_id, dis_shortest])
    
    pair_distance = np.array(dis_arr) 
    
    return pair_distance


def get_all_connected_paths(target_path_id, df, all_paths):
    
    connected_path_ids = []
    poc_connected_arr = []
    
    for path_id in all_paths.keys():
        subset_paths = all_paths.copy()
        del subset_paths[path_id]
        output_path_id, poc_connected = get_the_closest_path(point=all_paths[path_id][0], all_paths=subset_paths)
#         print(path_id, output_path_id, poc_connected)
        if output_path_id == target_path_id:
            connected_path_ids.append(path_id)
            poc_connected_arr.append(poc_connected) 
            
    return connected_path_ids, poc_connected_arr

def get_length_on_trace(point, path_id, all_paths):
    
    """
    Measure the length (or distance) from a certain point to the end of a path.
    """
    # print("measuring the distance from point ({}) to the end of path {}".format(point, path_id))
    trace = all_paths[path_id]
    boolean_arr = np.array(point == trace)
    for i, boolean in enumerate(boolean_arr):
        if (boolean).all():
            poc_id = i
            # print(i)
            
    length = np.sum(get_distance(trace[:poc_id][1:], trace[:poc_id][:-1]))
    
    return length

def find_trace_to_soma(ROI, soma_coords, df, threshold):

    """
    Find the trace from ROI to soma.

    Parameters
    ==========

    Outputs
    =======
    closest_path_id_arr:
        The array contains IDs of the shorest path from ROI to Soma. 

    poc_closest_path_arr:
        The array contains the "point of connection"(poc) where two paths connected with each other, 
        and those paths are the paths in ` closest_path_id_arr`.

    connected_paths_id_arr:
        The array contains all paths connected to a certain paths along the way from ROI to soma.

    poc_connected_paths_arr:
        The array contains all point of connection from all connected paths. 
    
    connected_paths_id_arr_short:
        The array contains the IDs of paths which are in the way between the ROI and soma.
    """
    
    point = ROI.copy() 
    all_paths = get_all_paths(df)
    
    closest_path_id_arr = []
    poc_closest_path_arr  = [] # poc: point of connection
    
    connected_paths_id_arr = []
    connected_paths_id_arr_short = []
    
    poc_connected_paths_arr = []
    
#     poc_connected_paths_processed_arr = []
    
    i = 0 # counter to stop while loop
    
    while not connected_with_soma(point, soma_coords, threshold) and i < 100:
        
        closest_path_id, poc_closest_path = get_the_closest_path(point, all_paths)    
        all_connected_path_ids, poc_connected_paths = get_all_connected_paths(closest_path_id, df, all_paths)
        point = all_paths[closest_path_id][0]
        print("the {}nd closest path is {}".format(i+1, closest_path_id))

        if len(poc_connected_paths) == 0:
            
            # print('Path {} has no connected paths.'.format(closest_path_id))
            
            closest_path_id_arr.append(closest_path_id)
            poc_closest_path_arr.append(poc_closest_path)
            poc_connected_paths_arr.append(poc_closest_path)
            
            all_paths.pop(closest_path_id)
            # print('\tDelete path {} in the remaining path list.'.format(closest_path_id))
        
        else:
            
            # print('Path {} has {} connected paths: {}'.format(closest_path_id, len(all_connected_path_ids), all_connected_path_ids))
            
            closest_path_id_arr.append(closest_path_id)
            poc_closest_path_arr.append(poc_closest_path)
            poc_connected_paths_arr.append(poc_closest_path)
            
            connected_paths_id_arr.append(all_connected_path_ids)
            
            
            trace_length_poc_closest = get_length_on_trace(poc_closest_path, path_id=closest_path_id, all_paths=get_all_paths(df))
            
            path_poc_dict = dict(zip(all_connected_path_ids, poc_connected_paths))

            
            for path_id in path_poc_dict.keys():
                trace_length_poc_connected = get_length_on_trace(path_poc_dict[path_id], all_paths=get_all_paths(df), path_id=closest_path_id)
                # print(trace_length_poc_closest, trace_length_poc_connected)
                if trace_length_poc_closest < trace_length_poc_connected:
                    continue
                else:
                    connected_paths_id_arr_short.append(path_id)
                    poc_connected_paths_arr.append(path_poc_dict[path_id])
            
            all_paths.pop(closest_path_id)
            # print('\tDelete path {} in the remaining path list.'.format(closest_path_id))
        
        i += 1 

    ########################
    ## Is this necessary? ##
    ########################
    # the_last_point = get_all_paths(df)[closest_path_id_arr[-1]][0]
    # poc_connected_paths_arr.append(the_last_point) # add the last point to the array
    # print(the_last_point)

    connected_paths_id_arr = sum(connected_paths_id_arr, [])
    print("the number of branches: ", len(poc_connected_paths_arr))    
        
    return closest_path_id_arr, poc_closest_path_arr, connected_paths_id_arr, poc_connected_paths_arr, connected_paths_id_arr_short


def distance_to_soma(point, soma_coords, df, threshold, debug=False):

    """
    calculate the distance between certain point and the soma along the neurite.
    """

    all_paths = get_all_paths(df)
    dis_arr = []
    i = 0
    while not connected_with_soma(point, soma_coords, threshold) and i < 100:
        
        
        path_id, poc = get_the_closest_path(point, all_paths)
        
        if debug:
            print(path_id, poc)
        
        dis = get_length_on_trace(poc, path_id, all_paths)
        dis_arr.append(dis)
        
        
        point = all_paths[path_id][0]
        all_paths.pop(path_id)
        
        i+=1
        
    return sum(dis_arr)

def check_intersection(pocs_ROI1, pocs_ROI2):
    
    """
    This one works. 
    """
    
    overlap_points = [poc1 for poc1 in pocs_ROI1 for poc2 in pocs_ROI2  if (poc1 == poc2).all()]
                
    if len(overlap_points) > 0:
        distance_arr = []
        for point in overlap_points:
            distance_arr.append(distance_to_soma(point, soma_coords, df, threshold))
        overlap_distance = np.max(distance_arr)
        inersected_point = overlap_points[np.argmax(distance_arr)]
        res = True
    else:
        res = False
        overlap_distance = 0
        
    # return res, overlap_distance, inersected_point
    return res, overlap_distance, overlap_points

def distance_between_two_ROIs(ROI1, ROI2, poc_processed1, poc_processed2, soma_coords, df, threshold):
    
    intersected, overlap_distance, overlap_points = check_intersection(poc_processed1, poc_processed2)
    distance_ROI1toSoma = distance_to_soma(ROI1, soma_coords, df, threshold)
    distance_ROI2toSoma = distance_to_soma(ROI2, soma_coords, df, threshold)    
    
    if intersected:
        distance_between_two_ROIs = distance_ROI1toSoma + distance_ROI2toSoma - overlap_distance
        number_of_branching = len(poc_processed1) + len(poc_processed2) - len(overlap_points) - 2
    
    else:
        
        distance_between_two_ROIs = distance_ROI1toSoma + distance_ROI2toSoma
        number_of_branching = len(poc_processed1) + len(poc_processed2) - 1
        
    return distance_between_two_ROIs, number_of_branching