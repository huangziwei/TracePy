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

def get_all_connected_paths(target_path_id, df, all_paths, threshold):
    """
    To Do
    =====
        return the coordinations of the closest points.
    """
    
    # first we need to delect the target_path from all_paths_list
    
    target_path = all_paths[target_path_id]
    pair = get_pair_distance_from_multiple_points_to_one_paths(target_path, all_paths) 
    res = pair[np.where([pair[: , 1] < threshold])[1]][:, 0].astype(int)
    
    connected_path_ids = np.delete(res, np.where(target_path_id == res)[0])
    
    poc_connected_arr = []
#     poc_path_id_arr = []
    for connected_path_id in connected_path_ids:
        
        path_coords = get_coords(df, connected_path_id)
        all_paths.pop(connected_path_id)
        _, poc_connected = get_the_closest_path(path_coords[0], all_paths)
#         poc_path_id_arr.append(path_id)
        poc_connected_arr.append(poc_connected)
        
    return connected_path_ids, poc_connected_arr


def get_length_on_trace(point, path_id, all_paths):
    
    trace = all_paths[path_id]
    boolean_arr = np.array(point == trace)
    for i, boolean in enumerate(boolean_arr):
        if (boolean).all():
            poc_id = i
#             print(i)
            
    length = np.sum(get_distance(trace[:poc_id][1:], trace[:poc_id][:-1]))
    
    return length

def find_trace_to_soma(ROI, soma_coords, df, threshold):
    
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
        all_connected_path_ids, poc_connected_paths = get_all_connected_paths(closest_path_id, df, all_paths, threshold)
        point = all_paths[closest_path_id][0]
        print("the {}nd closest path is {}".format(i+1, closest_path_id))

        if len(poc_connected_paths) == 0:
            
            print('Path {} has no connected paths.'.format(closest_path_id))
            
            closest_path_id_arr.append(closest_path_id)
            poc_closest_path_arr.append(poc_closest_path)
            poc_connected_paths_arr.append(poc_closest_path)
            
            all_paths.pop(closest_path_id)
        
        else:
            
            print('Path {} has {} connected paths: {}'.format(closest_path_id, len(all_connected_path_ids), all_connected_path_ids))
            
            closest_path_id_arr.append(closest_path_id)
            poc_closest_path_arr.append(poc_closest_path)
            poc_connected_paths_arr.append(poc_closest_path)
            
            connected_paths_id_arr.append(all_connected_path_ids.tolist())
            
            
            trace_length_poc_closest = get_length_on_trace(poc_closest_path, path_id=closest_path_id, all_paths=get_all_paths(df))
            
            path_poc_dict = dict(zip(all_connected_path_ids, poc_connected_paths))
            
            all_paths.pop(closest_path_id)

            for path_id in path_poc_dict.keys():
                trace_length_poc_connected = get_length_on_trace(path_poc_dict[path_id], all_paths=get_all_paths(df), path_id=closest_path_id)
                # print(trace_length_poc_closest, trace_length_poc_connected)
                if trace_length_poc_closest < trace_length_poc_connected:
                    continue
                else:
                    connected_paths_id_arr_short.append(path_id)
                    poc_connected_paths_arr.append(path_poc_dict[path_id])
        
        i += 1 
    connected_paths_id_arr = sum(connected_paths_id_arr, [])
    print("the number of branches: ", len(poc_connected_paths_arr))    
        
    return closest_path_id_arr, poc_closest_path_arr, connected_paths_id_arr, poc_connected_paths_arr, connected_paths_id_arr_short