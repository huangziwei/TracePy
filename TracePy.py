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

def get_coords(df, idx):
    
    x = df[df['id'] == str(idx)]['x'].astype('int').tolist()
    y = df[df['id'] == str(idx)]['y'].astype('int').tolist()
    z = df[df['id'] == str(idx)]['z'].astype('int').tolist()
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


# def get_branches_from_spheresoma(df, theta):
    
#     '''
#     :theta: scale factor of the soma radius
#     '''

#     path_dict = {}

#     ids = np.unique(df['id'].tolist())

#     for idx in ids:

#         path_dict[int(idx)] = get_coords(df, idx)
    
#     branches_from_soma = []
#     for key in path_dict.keys():

#         dis = np.sqrt(np.sum((path_dict[key][0] - np.roll(soma.centroid, 2)) ** 2))
#         if dis < theta * soma.radius * 512 / 340.48:
#             branches_from_soma.append(key)
#         else:
#             continue
            
#     return branches_from_soma

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