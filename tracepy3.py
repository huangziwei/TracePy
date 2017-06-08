from tp_utils import *

__all__ = ['TracingReg']

class TracingReg(object):

    def __init__(self, meta_exp):

        experimenter = meta_exp['experimenter']
        expdate = meta_exp['expdate']
        self.experimenter = experimenter
        self.expdate = expdate
        self.data_paths, self.celldir, self.exp_num = get_data_paths(experimenter, expdate)

        trace_path = self.data_paths['trace_path']
        stack_h5_path = self.data_paths['stack_h5_path']
        soma_h5_path = self.data_paths['soma_h5_path']

        self.cell_id = soma_h5_path.split('/')[7]
        self.df_trace, self.meta_trace = read_trace(trace_path)
        self.data_stack = load_h5_data(stack_h5_path)
        self.data_soma = load_h5_data(soma_h5_path)
        self.info_soma = get_info_soma(self.data_stack['wDataCh0'])

        print('Loaded necessary data.')

        self.stack_pixel_size = get_pixel_size_stack(self.data_stack)

        Local_tmp_path = '../Local_tmp/' + self.experimenter + '/' + self.expdate + '/' + self.exp_num
        if not os.path.exists(Local_tmp_path):
            os.makedirs(Local_tmp_path)

        if 'df_paths.pickle' in os.listdir(Local_tmp_path):
            print('\ndf_paths is loaded from existing pickle file.')
            self.df_paths = pd.read_pickle(Local_tmp_path + '/df_paths.pickle')
        else:
            print('\ndf_paths is created from df_trace, quality check is required. \nPlease run .check_paths_quality() and update_df_paths() \nto fix the wrong-directed paths.')
            self.df_paths = get_df_paths(self.df_trace)

        if 'df_rois.pickle' in os.listdir(Local_tmp_path):
            print('\ndf_rois is loaded from existing pickle file.')
            self.df_rois = pd.read_pickle(Local_tmp_path + '/df_rois.pickle')

        if 'df_roipairs.pickle' in os.listdir(Local_tmp_path):
            print('\ndf_roipairs is loaded from existing pickle file.')
            self.df_roipairs = pd.read_pickle(Local_tmp_path + '/df_roipairs.pickle')
    
    def check_paths_quality(self):

        df_paths = self.df_paths.copy()
        plt.figure(figsize=(20,20))
        for path_id in df_paths.path_id:
            path = df_paths.loc[path_id].path
            plt.plot(path[:, 1], path[:, 0])
            plt.scatter(path[0][1], path[0][0], color='red', s=140)
            plt.annotate(path_id, xy=(path[0][1]-2, path[0][0]-2), color='white', fontsize=10)
        plt.grid('off')

    def check_paths_updated_quality(self):

        df_paths = self.df_paths.copy()
        plt.figure(figsize=(20,20))
        for path_id in df_paths.path_id:
            path = df_paths.loc[path_id].path_updated
            plt.plot(path[:, 1], path[:, 0])
            plt.scatter(path[0][1], path[0][0], color='red', s=140)
            plt.annotate(path_id, xy=(path[0][1]-2, path[0][0]-2), color='white', fontsize=10)
        plt.grid('off')
    
    def fix_df_paths(self, path_ids):

        df_paths = self.df_paths.copy()
        for path_id in path_ids:
            new_path = df_paths.loc[path_id].path[::-1]
            df_paths.set_value(path_id, 'path', new_path)

        self.df_paths = df_paths

        self.check_paths_quality()

    def update_df_paths_info(self):

        df_paths = self.df_paths.copy()

        all_paths = df_paths.path.to_dict()

        path_updated = {}
        connect_to_dict = {}
        connect_to_at_dict = {}

        for i, key in enumerate(all_paths.keys()):

            if connected_with_soma(all_paths, key, self.info_soma, threshold=5):
                connect_to_dict[key] = -1
                connect_to_at_dict[key] = []
            else:
                connect_to_dict[key], connect_to_at_dict[key] = get_connect_to(all_paths, key)

            if connect_to_dict[key] == -1:
                path_updated[key] = all_paths[key]
            else:
                if (connect_to_at_dict[key] == all_paths[key]).all(1).any():
                    path_updated[key] = all_paths[key]
                else:
                    new_path = np.vstack([connect_to_at_dict[key], all_paths[key]])
                    path_updated[key] = new_path

        df_paths['path_updated'] = pd.Series(path_updated)
        df_paths['connect_to'] = pd.Series(connect_to_dict)
        df_paths['connect_to_at'] = pd.Series(connect_to_at_dict)

        connected_by_dict = {}
        connected_by_at_dict = {}

        for i, key in enumerate(df_paths.path_id):
            
            connected_by_dict[key]    = df_paths[df_paths.connect_to == key].path_id.tolist()
            connected_by_at_dict[key] = df_paths[df_paths.connect_to == key].connect_to_at.tolist()

        df_paths['connected_by'] = pd.Series(connected_by_dict)
        df_paths['connected_by_at'] = pd.Series(connected_by_at_dict)

        self.df_paths = df_paths.copy()

    def save_df_paths(self):

        save_dir = '../Local_tmp/' + self.experimenter + '/' + self.expdate + '/' + self.exp_num  

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.df_paths.to_pickle(save_dir + '/df_paths.pickle')


    def check_rois_on_trace(self, savefig=True):

        dendrites_h5_paths = self.data_paths['dendrites_h5_paths']

        (stack_soma_cx, stack_soma_cy, stack_soma_cz) = self.info_soma['centroid']
        linestack = trace2linestack(self.df_trace, self.meta_trace)
        linestack_xy = linestack.mean(2)
        
        df_rois = pd.DataFrame(columns=('recording_id', 
                                   'roi_id', 
                                   'recording_center', 
                                   'roi_coords_stack_xy',
                                   'recording_path'
                                   ))

        idx = 0
        rec_id = 0
        roi_id = 0

        for dendrite_h5_path in dendrites_h5_paths:

            dname = dendrite_h5_path.split('/')[-1].split('_')[2]

            d = load_h5_data(dendrite_h5_path)
            d_rec_rot = rotate_rec(d, self.data_stack)
            d_rois_rot, roi_coords_rot = rotate_roi(d, self.data_stack)

            d_rel_cy, d_rel_cx, d_rel_cz = rel_position_um(self.data_soma, d) / self.stack_pixel_size
            d_stack_cx, d_stack_cy = int(stack_soma_cx+d_rel_cx), int(stack_soma_cy+d_rel_cy) 
        
            padding = int(max(d_rec_rot.shape)) 
            crop = linestack_xy[d_stack_cx-padding:d_stack_cx+padding, d_stack_cy-padding:d_stack_cy+padding]

            scale_down = 0.9
            while 0 in np.unique(crop.shape):
                padding = int(scale_down * max(d_rec_rot.shape))
                crop = linestack_xy[d_stack_cx-padding:d_stack_cx+padding, d_stack_cy-padding:d_stack_cy+padding]
                scale_down *= scale_down

            d_rec_rot_x0, d_rec_rot_y0 = roi_matching(crop, d_rec_rot) # coords of roi in crop
            roi_coords_crop = roi_coords_rot + np.array([d_rec_rot_x0, d_rec_rot_y0])
            rec_center_crop = np.array([d_rec_rot.shape[0]/2,  d_rec_rot.shape[1]/2]) + np.array([d_rec_rot_x0, d_rec_rot_y0])

            roi_coords_stack_xy = roi_coords_crop + np.array([d_stack_cx-padding, d_stack_cy-padding])
            rec_center_stack_xy = rec_center_crop + np.array([d_stack_cx-padding, d_stack_cy-padding])


            ###################################
            ## Plot and check ROIs on traces ##
            ################################### 

            plt.figure(figsize=(32*3/5,32))

            ax1 = plt.subplot2grid((5,3), (0,0), rowspan=2, colspan=1)
            ax2 = plt.subplot2grid((5,3), (0,1), rowspan=2, colspan=2)
            ax3 = plt.subplot2grid((5,3), (2,0), rowspan=3, colspan=3)

            ax1.imshow(d_rec_rot, origin='lower')
            ax1.scatter(roi_coords_rot[:, 1], roi_coords_rot[:, 0], color='orange', s=80)

            ax2.imshow(crop, origin='lower')
            h_d_rec_rot, w_d_rec_rot = d_rec_rot.shape
            rect_d_rec_rot = plt.Rectangle((d_rec_rot_y0, d_rec_rot_x0), w_d_rec_rot, h_d_rec_rot , edgecolor='r', facecolor='none', linewidth=2)
            ax2.add_patch(rect_d_rec_rot)
            ax2.scatter(roi_coords_crop[:, 1], roi_coords_crop[:, 0], s=80, color='orange')

            ax3.imshow(linestack.mean(2), origin='lower', cmap=plt.cm.binary_r)
            hd, wd = crop.shape
            rect_crop = plt.Rectangle((d_stack_cy-padding, d_stack_cx-padding), wd, hd, edgecolor='r', facecolor='none', linewidth=2)
            h_d_rec_rot, w_d_rec_rot = d_rec_rot.shape
            rect_crop_d_rec = plt.Rectangle((d_rec_rot_y0 + d_stack_cy-padding, d_rec_rot_x0 + d_stack_cx-padding), w_d_rec_rot, h_d_rec_rot, edgecolor='r', facecolor='none', linewidth=2)
            ax3.add_patch(rect_crop_d_rec)
            ax3.add_patch(rect_crop)
            ax3.scatter(roi_coords_crop[:, 1]+d_stack_cy-padding, roi_coords_crop[:, 0]+d_stack_cx-padding, s=80, color='orange')
            ax3.annotate(dname, xy=(d_rec_rot_y0 + d_stack_cy-padding-10, d_rec_rot_x0 + d_stack_cx-padding-10), color='white')

            plt.suptitle('{}: {}'.format(self.expdate, dname))

            plt.grid('off')
            
            img_save_path = 'offline-data-check/' + self.expdate + '/' + self.cell_id
            
            if not os.path.exists(img_save_path):
                os.makedirs(img_save_path)

            if savefig:
                plt.savefig(img_save_path + '/{}-{}.png'.format(self.expdate,dname))

            ##########

            d_coords_xy = np.round(roi_coords_stack_xy).astype(int)
            # d_coords_xyz = np.zeros([d_coords_xy.shape[0], d_coords_xy.shape[1]+1])

            for i, roi_xy in enumerate(d_coords_xy):

                df_rois.loc[int(idx)] = [int(rec_id), int(roi_id), rec_center_stack_xy, roi_xy, dendrite_h5_path.split('/')[-1]]

                idx += 1
                roi_id += 1

            roi_id= 0
            rec_id += 1

        self.df_rois = df_rois

    def calibrate_rois(self):

        recording_ids = np.unique(self.df_rois['recording_id'])

        roi_coords_dict = {}
        Triggervalues_dict = {}
        Triggertimes_dict = {}
        Tracetimes0_dict = {}
        Traces0_raw_dict = {}
        rf_s_dict = {}
        rf_t_dict = {}
        path_id_dict = {}

        for rec_id in recording_ids:

            df_sub = self.df_rois[self.df_rois['recording_id'] == rec_id]

            recording_path = np.unique(df_sub['recording_path'])[0]

            d = load_h5_data(self.celldir + '/Pre/' + recording_path)
            linestack = trace2linestack(self.df_trace, self.meta_trace)

            for row in df_sub.iterrows():

                idx = row[0] # dataframe index

                roi_id = int(row[1]['roi_id'])
                roi_xy = row[1]['roi_coords_stack_xy']

                x_o = roi_xy[0] # roi_x_original
                y_o = roi_xy[1] # roi_y_original

                search_scope = 0
                offset = np.where(linestack[x_o:x_o+1, y_o:y_o+1] == 1)
                
                while offset[2].size == 0:
                    search_scope +=1
                    offset = np.where(linestack[x_o-search_scope:x_o+search_scope+1, y_o-search_scope:y_o+search_scope+1] == 1)
                
                z_o = np.mean(offset[2]).astype(int)  # roi_z_original, this is a guess

                x_c = np.arange(x_o-search_scope,x_o+search_scope+1)[offset[0]]
                y_c = np.arange(y_o-search_scope,y_o+search_scope+1)[offset[1]]
                z_c = offset[2]
                
                candidates = np.array([np.array([x_c[i], y_c[i], z_c[i]]) for i in range(len(x_c))])
                origins = np.array([x_o, y_o, z_o])
                
                x, y, z = candidates[np.argmin(np.sum((candidates - origins) ** 2, 1))]

                roi_coords_dict[idx] = np.array([x,y,z])

                path_id_dict[idx] = point_on_which_path(self.df_trace, np.array([x,y,z]))
                
                Triggervalues_dict[idx] = d['Triggervalues']
                Triggertimes_dict[idx] = d['Triggertimes']
                Tracetimes0_dict[idx] = d['Tracetimes0'][:, roi_id]
                Traces0_raw_dict[idx] = d['Traces0_raw'][:, roi_id]
                rf_s_dict[idx] = d['STRF_SVD_Space0'][:, roi_id]
                rf_t_dict[idx] = d['STRF_SVD_Time0'][:, roi_id]

        self.df_rois['roi_coords'] = pd.Series(roi_coords_dict)
        self.df_rois['path_id'] = pd.Series(path_id_dict)
        self.df_rois['Triggervalues'] = pd.Series(Triggervalues_dict)
        self.df_rois['Triggertimes'] = pd.Series(Triggertimes_dict)
        self.df_rois['Tracetimes0'] = pd.Series(Tracetimes0_dict)
        self.df_rois['Traces0_raw'] = pd.Series(Traces0_raw_dict)
        self.df_rois['rf_s'] = pd.Series(rf_s_dict)
        self.df_rois['rf_t'] = pd.Series(rf_t_dict)


    def plot_all_rois(self):

        linestack = trace2linestack(self.df_trace, self.meta_trace)

        plt.figure(figsize=(16,16))
        plt.imshow(linestack.sum(2), origin='lower', cmap=plt.cm.binary)

        plt.imshow(self.info_soma['mask_2d'], origin='lower', cmap=plt.cm.binary, vmin=0.0, alpha=0.3)

        rois_pos = np.vstack(self.df_rois.roi_coords)
        rois_dis = self.df_rois.distance_dendritic.values
        plt.scatter(rois_pos[:, 1], rois_pos[:, 0], c=rois_dis/self.df_rois.distance_dendritic.max(), s=180, cmap=plt.cm.viridis)
        
        for i, roi_id in enumerate(self.df_rois.index):
            plt.annotate(int(roi_id), xy=(rois_pos[i][1]-3, rois_pos[i][0]-3), color='white', fontsize=16*0.7)
        scalebar = ScaleBar(self.stack_pixel_size, units='um', location='upper left', box_alpha=0)
        plt.gca().add_artist(scalebar)

    def update_df_rois_with_segments_and_branchpoints(self):

        df_rois = self.df_rois.copy()
        df_paths = self.df_paths.copy()

        branchpoints_dict = {}
        segments_dict = {}
        distance_dendritic_dict = {}
        distance_radial_dict = {}

        for row in df_rois.iterrows():

            idx = row[0] # dataframe index
            roi_id = int(row[1]['roi_id'])
            path_id = int(row[1]['path_id'])
            roi_pos = row[1]['roi_coords']

            # connected_points = df_paths.loc[path_id].connected_by_at

            sms, bpts = get_all_segments_and_branchpoints(roi_pos, path_id, df_paths)

            if len(bpts) >=1 and bpts != [[]]:

                bpts = np.vstack([p for p in bpts if p != []])

            if len(sms) >1:
                sm = np.vstack(sms[::-1]) # stack all segments into one
            else:
                sm =sms[0]

            branchpoints_dict[idx] = bpts
            segments_dict[idx] = sm
            distance_dendritic_dict[idx] = np.sum(np.sqrt(np.sum((sm[1:] - sm[:-1])**2, 1))) * self.stack_pixel_size
            distance_radial_dict[idx] = np.sqrt(np.sum((self.info_soma['centroid'] - df_rois.loc[roi_id].roi_coords) ** 2)) * self.stack_pixel_size
        
        df_rois['branchpoints'] = pd.Series(branchpoints_dict)
        df_rois['segments'] = pd.Series(segments_dict)
        df_rois['distance_dendritic'] = pd.Series(distance_dendritic_dict)
        df_rois['distance_radial'] = pd.Series(distance_radial_dict)
        
        self.df_rois = df_rois.copy()

    def plot_roi2soma(self):

        df_rois = self.df_rois.copy()

        for row in df_rois.iterrows():
            
            idx = row[0]
            rec_id = row[1]['recording_id']
            roi_id = row[1]['roi_id']
            
            linestack = trace2linestack(self.df_trace, self.meta_trace)
            
            plt.figure(figsize=(16,16))
            plt.imshow(linestack.sum(2), origin='lower', cmap=plt.cm.binary) # Cell Skeleton
            plt.imshow(self.info_soma['mask_2d'], origin='lower', cmap=plt.cm.binary, vmin=0.0, alpha=0.3) # Soma
            plt.grid('off')

            all_rois_pos = np.vstack(df_rois.roi_coords) # all rois
            plt.scatter(all_rois_pos[:, 1], all_rois_pos[:, 0], c='gray', s=180, alpha=0.3)

            sm = self.df_rois.loc[idx].segments # dendritic trace from roi to soma
            plt.plot(sm[:, 1], sm[:, 0], lw=2.5, color='red')

            bpts = df_rois.loc[idx].branchpoints
            roi_pos = df_rois.loc[idx].roi_coords

            plt.scatter(bpts[:, 1], bpts[:, 0], color='black', s=80, zorder=10) # bpts from roi to soma
            plt.scatter(roi_pos[1], roi_pos[0], color='red', s=180) # roi of interest

            scalebar = ScaleBar(self.stack_pixel_size, units='um', location='upper left', box_alpha=0)
            plt.gca().add_artist(scalebar)
            
            plt.title('rec {}: roi {}'.format(rec_id, roi_id))
            
            img_save_path = 'img-roi2soma/' + self.expdate + '/' + self.exp_num
            if not os.path.exists(img_save_path):
                    os.makedirs(img_save_path) 
            
            plt.savefig(img_save_path + '/{}-{}-{}.png'.format(idx, int(rec_id), int(roi_id)))


    def get_roi_pairs(self):

        df_roipairs = pd.DataFrame(columns=('pair_id',
                                        'segments_between',
                                        'branchpoints_between'))

        indices = np.array(self.df_rois.index)

        i = 0
        for roi_id0 in indices:

            indices = np.delete(indices, 0) # delete the first values in every loop

            for roi_id1 in indices:
                print('Processing pair ({} {})...'.format(roi_id0, roi_id1))
                sms_between, bpts_between = get_info_roi2roi(self.df_trace, self.df_paths, self.df_rois, self.info_soma, roi_id0, roi_id1)
                df_roipairs.loc[i] = [set([roi_id0, roi_id1]), sms_between, bpts_between]
                i+= 1


        self.df_roipairs = df_roipairs