from tp_utils import *
from sta_utils import *

__all__ = ['TracingReg']

class TracingReg(object):

    def __init__(self, meta_exp):

        print('Loading data...')

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

        self.linestack = trace2linestack(self.df_trace, self.meta_trace)
        self.stack_pixel_size = get_pixel_size_stack(self.data_stack)

        self.IMT_Data_path = IMT_Data_path = '../IMT_Data/' + self.experimenter + '/' + self.expdate + '/' + self.exp_num
        if not os.path.exists(IMT_Data_path):
            os.makedirs(IMT_Data_path)

        if 'info_soma_updated.pickle' in os.listdir(IMT_Data_path):
            print('\ninfo_soma_updated is loaded from existing pickle file.')

            with open(IMT_Data_path + '/info_soma_updated.pickle', 'rb') as input_file:
                self.info_soma = pickle.load(input_file)
        elif 'info_soma.pickle' in os.listdir(IMT_Data_path):
            print('\ninfo_soma is loaded from existing pickle file.')

            with open(IMT_Data_path + '/info_soma.pickle', 'rb') as input_file:
                self.info_soma = pickle.load(input_file)
        else:
            print('\nReading info_soma from data_stack.')
            
            try:
                stack = self.data_stack['wDataCh0_warped'].copy()
                stack[stack == 65535] = stack[stack != 65535].mean()
                # stack[stack == 65535] = 0
            except KeyError:
                stack = self.data_stack['wDataCh0'].copy()
            
            if 'adjust' in meta_exp.keys():
                info_soma = get_info_soma(stack, meta_exp['adjust'])
            else:
                info_soma = get_info_soma(stack)

            # with open(IMT_Data_path + '/info_soma.pickle', 'wb') as output_file:
            #     pickle.dump(info_soma, output_file)

            self.info_soma = info_soma

        if 'df_paths.pickle' in os.listdir(IMT_Data_path):
            print('\ndf_paths is loaded from existing pickle file.')
            self.df_paths = pd.read_pickle(IMT_Data_path + '/df_paths.pickle')
        else:
            print('\ndf_paths is created from df_trace, quality check is required. \nPlease run .check_paths_quality() and update_df_paths() \nto fix the wrong-directed paths.')
            self.df_paths = get_df_paths(self.df_trace)

        if 'df_rois_updated.pickle' in os.listdir(IMT_Data_path):
            print('\ndf_rois_updated is loaded from existing pickle file.')
            self.df_rois = pd.read_pickle(IMT_Data_path + '/df_rois_updated.pickle')
        elif 'df_rois.pickle' in os.listdir(IMT_Data_path):
            print('\ndf_rois is loaded from existing pickle file.')
            self.df_rois = pd.read_pickle(IMT_Data_path + '/df_rois.pickle')
        else:
            print('\nPlease scroll down to df_rois session.')

        if 'df_roipairs.pickle' in os.listdir(IMT_Data_path):
            print('\ndf_roipairs is loaded from existing pickle file.')
            self.df_roipairs = pd.read_pickle(IMT_Data_path + '/df_roipairs.pickle')
        else:
            print('\nPlease scroll down to df_roipairs session.')

    def save_info_soma(self):
        print('Saving info_soma to folder [{}]'.format(self.IMT_Data_path))
        with open(self.IMT_Data_path + '/info_soma.pickle', 'wb') as output_file:
            pickle.dump(self.info_soma, output_file)
    



    def _three_views(self, plot_rois=False, plot_bands=False, colorbar=False, roi_point_size=80, roi_max_distance=350):

        plt.figure(figsize=(16,16))

        ax1 = plt.subplot2grid((4,4), (0,1), rowspan=3, colspan=3)
        ax2 = plt.subplot2grid((4,4), (0,0), rowspan=3, colspan=1)
        ax3 = plt.subplot2grid((4,4), (3,1), rowspan=1, colspan=3)
        ax4 = plt.subplot2grid((4,4), (3,0), rowspan=1, colspan=1)

        linestack = trace2linestack(self.df_trace, self.meta_trace)

        linestack_xy = linestack.sum(2)
        linestack_xy[linestack_xy != 0] = 1

        linestack_xz = linestack.sum(1)
        linestack_xz[linestack_xz != 0] = 1

        linestack_yz = linestack.sum(0)
        linestack_yz[linestack_yz != 0] = 1

        soma_centroid = self.info_soma['centroid']
        mask_xy =  self.info_soma['mask_xy']
        mask_xz =  self.info_soma['mask_xz']
        mask_yz =  self.info_soma['mask_yz']

        # sideview (right to left)

        ax2.imshow(linestack_xz, origin='lower', cmap=plt.cm.binary)
        ax2.imshow(mask_xz, origin='lower', cmap=plt.cm.binary, vmin=0.0, alpha=0.3)
        ax2.scatter(soma_centroid[2], soma_centroid[0], color='red')

        ax2.axis('off')

        # sideview (left to right)
        ax3.imshow(linestack_yz.T, origin='lower', cmap=plt.cm.binary)
        ax3.imshow(mask_yz.T, origin='lower', cmap=plt.cm.binary, vmin=0.0, alpha=0.3)
        ax3.scatter(soma_centroid[1], soma_centroid[2], color='red')
        ax3.axis('off')

        # empty box
        ax4.axis('off')
        # topview

        ax1.imshow(linestack_xy, origin='lower', cmap=plt.cm.binary)
        ax1.imshow(mask_xy, origin='lower', cmap=plt.cm.binary, vmin=0.0, alpha=0.3)
        ax1.scatter(soma_centroid[1], soma_centroid[0], color='red')

        ax1.axis('off')
        scalebar = ScaleBar(self.stack_pixel_size, units='um', location='lower left', box_alpha=0, pad=4)
        ax1.add_artist(scalebar)    
        
        if plot_rois:
            
            rois_pos = np.vstack(self.df_rois.roi_coords)
            rois_dis = self.df_rois.distance_dendritic.values
            sc = ax1.scatter(rois_pos[:, 1], rois_pos[:, 0], c=rois_dis, s=roi_point_size, cmap=plt.cm.viridis, vmin=0, vmax=roi_max_distance)
            if colorbar:
                cbar = plt.colorbar(sc, ax=ax1, fraction=0.02, pad=-.05 )
                cbar.outline.set_visible(False)

            ax2.scatter(rois_pos[:, 2], rois_pos[:, 0], c=rois_dis, s=roi_point_size * 0.8, cmap=plt.cm.viridis, vmin=0, vmax=roi_max_distance)
            ax3.scatter(rois_pos[:, 1], rois_pos[:, 2], c=rois_dis, s=roi_point_size * 0.8, cmap=plt.cm.viridis, vmin=0, vmax=roi_max_distance)

        if plot_bands:
            sumLine = self.data_stack['sumLine']
            sumLine /= sumLine.max()
            num_layers = len(sumLine)
            ON = int(input('Please enter the ON layer: '))
            OFF = int(input('Please enter the OFF layer: '))
            layerON  = (OFF - ON) * 0.48 + ON
            layerOFF =  (OFF - ON) * 0.77 + ON

            ax2.plot(np.arange(num_layers), sumLine * 30, color='black')  
            ax2.axvline(layerON, color='red', linestyle='dashed')
            ax2.axvline(layerOFF, color='red', linestyle='dashed')    
            ax2.annotate('ON', xy=(layerON, 0), xytext=(layerON-10, -10), zorder=10,weight="bold")
            ax2.annotate('OFF', xy=(layerOFF, 0), xytext=(layerOFF-10, -10),zorder=10, weight="bold")

            ax3.plot(sumLine * 30, np.arange(num_layers), color='black')
            ax3.axhline(layerON, color='red', linestyle='dashed')
            ax3.axhline(layerOFF, color='red', linestyle='dashed')
            ax3.annotate('ON', xy=(0, layerON), xytext=(-22, layerON-5), zorder=10,weight="bold")
            ax3.annotate('OFF', xy=(0, layerOFF), xytext=(-22, layerOFF-5),zorder=10, weight="bold")


    def check_soma_mask(self, plot_bands=False):
        self._three_views(plot_rois=False, plot_bands=plot_bands)

    def check_paths_quality(self):

        df_paths = self.df_paths.copy()
        plt.figure(figsize=(20,20))
        plt.imshow(self.info_soma['mask_xy'], origin='lower', cmap=plt.cm.binary, vmin=0.0, alpha=0.3) # Soma
        plt.scatter(self.info_soma['centroid'][1], self.info_soma['centroid'][0], color='red')
        
        for path_id in df_paths.path_id:
            path = df_paths.loc[path_id].path
            plt.plot(path[:, 1], path[:, 0])
            plt.scatter(path[0][1], path[0][0], color='red', s=140)
            plt.arrow(path[1][1], path[1][0], path[1][1] - path[2][1], path[1][0] - path[2][0],  fc='k', ec='k', width=1, head_width=5, head_length=5)
            plt.annotate(path_id, xy=(path[0][1]-2, path[0][0]-2), color='white', fontsize=8)
        
        plt.axis('off')
        scalebar = ScaleBar(self.stack_pixel_size, units='um', location='upper left', box_alpha=0)
        plt.gca().add_artist(scalebar)

    def check_paths_updated_quality(self):

        df_paths = self.df_paths.copy()
        plt.figure(figsize=(20,20))
        plt.imshow(self.info_soma['mask_xy'], origin='lower', cmap=plt.cm.binary, vmin=0.0, alpha=0.3) # Soma
        plt.scatter(self.info_soma['centroid'][1], self.info_soma['centroid'][0], color='red')
        
        for path_id in df_paths.path_id:
            path = df_paths.loc[path_id].path_updated
            plt.plot(path[:, 1], path[:, 0])
            plt.scatter(path[0][1], path[0][0], color='red', s=140)
            plt.arrow(path[1][1], path[1][0], path[1][1] - path[2][1], path[1][0] - path[2][0],  fc='k', ec='k', width=1, head_width=5, head_length=5)
            plt.annotate(path_id, xy=(path[0][1]-2, path[0][0]-2), color='white', fontsize=8)
 
        plt.axis('off')
        scalebar = ScaleBar(self.stack_pixel_size, units='um', location='upper left', box_alpha=0)
        plt.gca().add_artist(scalebar)
    
    def fix_df_paths(self, path_ids):
        if 'df_paths.pickle' in os.listdir(self.IMT_Data_path):
            print('The current df_paths is loaded from existing file, it means there\'s no need to fix it. ')
        else:
            df_paths = self.df_paths.copy()
            for path_id in path_ids:
                new_path = df_paths.loc[path_id].path[::-1]
                df_paths.set_value(path_id, 'path', new_path)

            self.df_paths = df_paths

            self.check_paths_quality()

    def update_df_paths_info(self, exception=None, threshold=3):

        """
        exception: dict
            a dictionary to handle exception cases. e.g.
            {key: to delete}
        """

        df_paths = self.df_paths.copy()

        all_paths = df_paths.path.to_dict()

        path_updated = {}
        connect_to_dict = {}
        connect_to_at_dict = {}

        for i, key in enumerate(all_paths.keys()):

            if connected_with_soma(all_paths, key, self.info_soma, threshold=threshold):
                print('Path {} is connected with soma'.format(key))
                connect_to_dict[key] = -1
                connect_to_at_dict[key] = []
            else:
                if exception != None and key in exception.keys():
                    connect_to_dict[key], connect_to_at_dict[key] = get_connect_to(all_paths, key, exception)
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

        # check if all paths can link back to soma
        for row in df_paths.iterrows():
    
            idx = row[0]
            path_id = row[1]['path_id']
            connect_to = row[1]['connect_to']
            paths2soma = [connect_to]
            
            while connect_to != -1:
                
                path_id = connect_to
                connect_to = df_paths.loc[path_id]['connect_to']

                if connect_to in paths2soma:
                    paths2soma.append(connect_to)
                    print("Path [{}] cannot trace back to soma: {}.".format(idx, paths2soma))
                    break
                else:
                    paths2soma.append(connect_to)

        connected_by_dict = {}
        connected_by_at_dict = {}

        for i, key in enumerate(df_paths.path_id):
            
            connected_by_dict[key]    = df_paths[df_paths.connect_to == key].path_id.tolist()
            connected_by_at_dict[key] = df_paths[df_paths.connect_to == key].connect_to_at.tolist()

        df_paths['connected_by'] = pd.Series(connected_by_dict)
        df_paths['connected_by_at'] = pd.Series(connected_by_at_dict)

        self.df_paths = df_paths.copy()

    # def save_df_paths(self):

    #     save_dir = '../IMT_Data/' + self.experimenter + '/' + self.expdate + '/' + self.exp_num  
    #     print('Saving df_paths to folder [{}]'.format(save_dir))
    #     if not os.path.exists(save_dir):
    #         os.makedirs(save_dir)

    #     self.df_paths.to_pickle(save_dir + '/df_paths.pickle')
    def save_df_paths(self):

        # save_dir = '../IMT_Data/' + self.experimenter + '/' + self.expdate + '/' + self.exp_num  

        print('Saving df_paths to folder [{}]'.format(self.IMT_Data_path))
        if not os.path.exists(self.IMT_Data_path):
            os.makedirs(self.IMT_Data_path)

        self.df_paths.to_pickle(self.IMT_Data_path + '/df_paths.pickle')

    def check_rois_on_trace(self, padding_param=1, savefig=True):

        import matplotlib as mpl

        dendrites_h5_paths = self.data_paths['dendrites_h5_paths']

        (stack_soma_cx, stack_soma_cy, stack_soma_cz) = self.info_soma['centroid']
        linestack = trace2linestack(self.df_trace, self.meta_trace)
        linestack_xy = linestack.mean(2)
        
        df_rois = pd.DataFrame(columns=('recording_id', 
                                   'roi_id', 
                                   'recording_center', 
                                   'roi_coords_stack_xy',
                                   'filename'
                                   ))

        idx = 0
        rec_id = 1
        roi_id = 1

        for dendrite_h5_path in dendrites_h5_paths:

            dname = dendrite_h5_path.split('/')[-1].split('_')[2]

            d = load_h5_data(dendrite_h5_path)
            d_rec = resize_rec(d, self.data_stack)
            d_rec_rot, (origin_shift_x, origin_shift_y) = rotate_rec(d, self.data_stack)
            d_rois_rot, roi_coords_rot = rotate_roi(d, self.data_stack)

            d_rel_cy, d_rel_cx, d_rel_cz = rel_position_um(self.data_soma, d) / self.stack_pixel_size
            d_stack_cx, d_stack_cy = int(stack_soma_cx+d_rel_cx), int(stack_soma_cy+d_rel_cy) 
        
            padding = int(max(d_rec_rot.shape) * padding_param) 

            crop = linestack_xy[d_stack_cx-padding:d_stack_cx+padding, d_stack_cy-padding:d_stack_cy+padding]

            scale_down = 0.9
            while 0 in np.unique(crop.shape):
                padding = int(scale_down * max(d_rec_rot.shape))
                crop = linestack_xy[d_stack_cx-padding:d_stack_cx+padding, d_stack_cy-padding:d_stack_cy+padding]
                scale_down *= scale_down

            d_rec_rot_x0, d_rec_rot_y0 = roi_matching(crop, d_rec_rot) # the origin of the rotated rec region in the crop region 
            
            roi_coords_crop = roi_coords_rot + np.array([d_rec_rot_x0, d_rec_rot_y0])
            d_rois_rot_crop = np.pad(d_rois_rot, pad_width=((d_rec_rot_x0, 0), (d_rec_rot_y0, 0)), mode='constant', constant_values=255)
            d_rois_rot_crop = np.ma.masked_where(d_rois_rot_crop == 255, d_rois_rot_crop)

            rec_center_crop = np.array([d_rec_rot.shape[0]/2,  d_rec_rot.shape[1]/2]) + np.array([d_rec_rot_x0, d_rec_rot_y0])

            roi_coords_stack_xy = roi_coords_crop + np.array([d_stack_cx-padding, d_stack_cy-padding])
            d_rois_rot_stack_xy = np.pad(d_rois_rot_crop, pad_width=((d_stack_cx-padding, 0), (d_stack_cy-padding, 0)), mode='constant', constant_values=255)
            d_rois_rot_stack_xy = np.ma.masked_where(d_rois_rot_stack_xy == 255, d_rois_rot_stack_xy)

            rec_center_stack_xy = rec_center_crop + np.array([d_stack_cx-padding, d_stack_cy-padding])


            ###################################
            ## Plot and check ROIs on traces ##
            ################################### 

            plt.figure(figsize=(32*3/5,32))

            ax1 = plt.subplot2grid((5,3), (0,0), rowspan=2, colspan=1)
            ax2 = plt.subplot2grid((5,3), (0,1), rowspan=2, colspan=2)
            ax3 = plt.subplot2grid((5,3), (2,0), rowspan=3, colspan=3)

            ax1.imshow(d_rec_rot, origin='lower')
            ax1.imshow(d_rois_rot, origin='lower', cmap=plt.cm.viridis)
            ax1.grid('off')
            ax1.scatter(roi_coords_rot[:, 1], roi_coords_rot[:, 0], color='orange', s=80)
            ax1.set_title('Recording Region', fontsize=24)

            ax2.imshow(crop, origin='lower')
            h_d_rec_rot, w_d_rec_rot = d_rec.shape
            rect_d_rec_rot = mpl.patches.Rectangle((d_rec_rot_y0+origin_shift_y, d_rec_rot_x0+origin_shift_x), w_d_rec_rot, h_d_rec_rot , edgecolor='r', facecolor='none', linewidth=2)
            tmp2 = mpl.transforms.Affine2D().rotate_deg_around(d_rec_rot_y0+origin_shift_y, d_rec_rot_x0+origin_shift_x, -d['wParamsNum'][31]) + ax2.transData
            rect_d_rec_rot.set_transform(tmp2)
            ax2.add_patch(rect_d_rec_rot)
            ax2.imshow(d_rois_rot_crop, origin='lower', cmap=plt.cm.viridis)
            ax2.scatter(roi_coords_crop[:, 1], roi_coords_crop[:, 0], s=80, color='orange')
            ax2.set_title('Cropped Region', fontsize=24)
            ax2.grid('off')

            ax3.imshow(linestack.mean(2), origin='lower', cmap=plt.cm.binary)
            ax3.imshow(self.info_soma['mask_xy'], origin='lower', cmap=plt.cm.binary, vmin=0.0, alpha=0.3)

            hd, wd = crop.shape
            rect_crop = mpl.patches.Rectangle((d_stack_cy-padding, d_stack_cx-padding), wd, hd, edgecolor='r', facecolor='none', linewidth=2)
            
            h_d_rec_rot, w_d_rec_rot = d_rec.shape
            rect_crop_d_rec = mpl.patches.Rectangle((d_rec_rot_y0 + d_stack_cy-padding+origin_shift_y, d_rec_rot_x0 + d_stack_cx-padding+origin_shift_x), w_d_rec_rot, h_d_rec_rot, edgecolor='r', facecolor='none', linewidth=2)
            tmp3 = mpl.transforms.Affine2D().rotate_deg_around(d_rec_rot_y0+ d_stack_cy-padding+origin_shift_y, d_rec_rot_x0+d_stack_cx-padding+origin_shift_x, -d['wParamsNum'][31]) + ax3.transData
            rect_crop_d_rec.set_transform(tmp3)
            
            ax3.add_patch(rect_crop_d_rec)
            ax3.add_patch(rect_crop)
            ax3.imshow(d_rois_rot_stack_xy, origin='lower', cmap=plt.cm.viridis)
            ax3.scatter(roi_coords_crop[:, 1]+d_stack_cy-padding, roi_coords_crop[:, 0]+d_stack_cx-padding, s=80, color='orange')
            ax3.annotate(dname, xy=(d_rec_rot_y0 + d_stack_cy-padding-10, d_rec_rot_x0 + d_stack_cx-padding-10), color='white')
            ax3.set_title('ROIs on Cell Morpholoy', fontsize=24)
            ax3.grid('off')

            scalebar = ScaleBar(self.stack_pixel_size, units='um', location='lower left', box_alpha=0, pad=4)
            ax3.add_artist(scalebar)
            
            plt.suptitle('{}: {}'.format(self.expdate, dname), fontsize=28)

            
            img_save_path = './Fig_roi_mapping/' + self.expdate + '/' + self.cell_id
            
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

            roi_id = 1
            rec_id += 1

        self.df_rois = df_rois

    def manually_adjust_roi_coord(self, rec_id, offset=[0,0], angle_adjust=0, padding_param=1, pad_by_hand=[0, 0, 0, 0], adjust=False, savefig=False):

        print("Use `padding_param=float` to scale the padding, or use `pad_by_hand=[bottom, top, left, right]` to pad specific edges.")

        idx_subframe = self.df_rois[self.df_rois['recording_id'] == rec_id]['recording_center'].index.tolist()

        dendrites_h5_paths = self.data_paths['dendrites_h5_paths']
        dendrite_h5_path = dendrites_h5_paths[rec_id-1]

        (stack_soma_cx, stack_soma_cy, stack_soma_cz) = self.info_soma['centroid']
        linestack = trace2linestack(self.df_trace, self.meta_trace)
        linestack_xy = linestack.mean(2)


        d = load_h5_data(dendrite_h5_path)
        d_rec = resize_rec(d, self.data_stack)
        d_rec_rot, (origin_shift_x, origin_shift_y) = rotate_rec(d, self.data_stack)
        d_rois_rot, roi_coords_rot = rotate_roi(d, self.data_stack, angle_adjust=angle_adjust)

        dname = dendrite_h5_path.split('/')[-1].split('_')[2]


        d_rel_cy, d_rel_cx, d_rel_cz = rel_position_um(self.data_soma, d) / self.stack_pixel_size
        d_stack_cx, d_stack_cy = int(stack_soma_cx+d_rel_cx), int(stack_soma_cy+d_rel_cy) 

        padding = int(max(d_rec_rot.shape) * padding_param) 
        padb, padt, padl, padr = pad_by_hand

        lim_bottom = d_stack_cx-padding-padb
        if lim_bottom < 0:
            lim_bottom = 0

        lim_top = d_stack_cx+padding+padt
        if lim_top > 512:
            lim_top = 511

        lim_left   = d_stack_cy-padding-padl
        if lim_left < 0:
            lim_left = 0

        lim_right  = d_stack_cy+padding+padr
        if lim_right > 512:
            lim_right = 511

        crop = linestack_xy[lim_bottom:lim_top, lim_left:lim_right]

        d_rec_rot_x0, d_rec_rot_y0 = roi_matching(crop, d_rec_rot) # the origin of the rotated rec region in the crop region 
        d_rec_rot_x0 += offset[0]
        d_rec_rot_y0 += offset[1] # update the origin of the rotated rec region in the crop region with the new offset

        roi_coords_crop = roi_coords_rot + np.array([d_rec_rot_x0, d_rec_rot_y0])
        d_rois_rot_crop = np.pad(d_rois_rot, pad_width=((d_rec_rot_x0, 0), (d_rec_rot_y0, 0)), mode='constant', constant_values=255)
        d_rois_rot_crop = np.ma.masked_where(d_rois_rot_crop == 255, d_rois_rot_crop)

        rec_center_crop = np.array([d_rec_rot.shape[0]/2,  d_rec_rot.shape[1]/2]) + np.array([d_rec_rot_x0, d_rec_rot_y0])

        roi_coords_stack_xy = roi_coords_crop + np.array([lim_bottom, lim_left])

        d_rois_rot_stack_xy = np.pad(d_rois_rot_crop, pad_width=((lim_bottom, 0), (lim_left, 0)), mode='constant', constant_values=255)
        d_rois_rot_stack_xy = np.ma.masked_where(d_rois_rot_stack_xy == 255, d_rois_rot_stack_xy)
        rec_center_stack_xy = rec_center_crop + np.array([lim_bottom, lim_left])


        ###################################
        ## Plot and check ROIs on traces ##
        ################################### 

        plt.figure(figsize=(32*3/5,32))

        ax1 = plt.subplot2grid((5,3), (0,0), rowspan=2, colspan=1)
        ax2 = plt.subplot2grid((5,3), (0,1), rowspan=2, colspan=2)
        ax3 = plt.subplot2grid((5,3), (2,0), rowspan=3, colspan=3)

        ax1.imshow(d_rec_rot, origin='lower')
        ax1.imshow(d_rois_rot, origin='lower', cmap=plt.cm.viridis)
        ax1.scatter(roi_coords_rot[:, 1], roi_coords_rot[:, 0], color='orange', s=80)
        ax1.set_title('Recording Region', fontsize=24)

        ax2.imshow(crop, origin='lower')
        h_d_rec_rot, w_d_rec_rot = d_rec.shape
        rect_d_rec_rot = mpl.patches.Rectangle((d_rec_rot_y0+origin_shift_y, d_rec_rot_x0+origin_shift_x), w_d_rec_rot, h_d_rec_rot , edgecolor='r', facecolor='none', linewidth=2)
        tmp2 = mpl.transforms.Affine2D().rotate_deg_around(d_rec_rot_y0+origin_shift_y, d_rec_rot_x0+origin_shift_x, -d['wParamsNum'][31]) + ax2.transData
        rect_d_rec_rot.set_transform(tmp2)

        ax2.add_patch(rect_d_rec_rot)
        ax2.imshow(d_rois_rot_crop, origin='lower', cmap=plt.cm.viridis)
        ax2.scatter(roi_coords_crop[:, 1], roi_coords_crop[:, 0], s=80, color='orange')
        ax2.set_title('Cropped Region', fontsize=24)

        ax3.imshow(linestack.mean(2), origin='lower', cmap=plt.cm.binary)
        ax3.imshow(self.info_soma['mask_xy'], origin='lower', cmap=plt.cm.binary, vmin=0.0, alpha=0.3)

        hd, wd = crop.shape
        rect_crop = mpl.patches.Rectangle((lim_left, lim_bottom), wd, hd, edgecolor='r', facecolor='none', linewidth=2)
        h_d_rec_rot, w_d_rec_rot = d_rec.shape
        rect_crop_d_rec = mpl.patches.Rectangle((d_rec_rot_y0 + lim_left+origin_shift_y, d_rec_rot_x0 + lim_bottom+origin_shift_x), w_d_rec_rot, h_d_rec_rot, edgecolor='r', facecolor='none', linewidth=2)
        tmp3 = mpl.transforms.Affine2D().rotate_deg_around(d_rec_rot_y0+ lim_left+origin_shift_y, d_rec_rot_x0+lim_bottom+origin_shift_x, -d['wParamsNum'][31]) + ax3.transData
        rect_crop_d_rec.set_transform(tmp3)

        ax3.add_patch(rect_crop_d_rec)
        ax3.add_patch(rect_crop)
        ax3.imshow(d_rois_rot_stack_xy, origin='lower', cmap=plt.cm.viridis)
        ax3.scatter(roi_coords_crop[:, 1]+lim_left, roi_coords_crop[:, 0]+lim_bottom, s=80, color='orange')
        ax3.annotate(dname, xy=(d_rec_rot_y0 + d_stack_cy-padding-10, d_rec_rot_x0 + d_stack_cx-padding-10), color='white')
        ax3.set_title('ROIs on Cell Morpholoy', fontsize=24)
        scalebar = ScaleBar(self.stack_pixel_size, units='um', location='lower left', box_alpha=0, pad=4)
        ax3.add_artist(scalebar)

        plt.suptitle('{}: {}'.format(self.expdate, dname), fontsize=28)

        plt.grid('off')

        if savefig:
            img_save_path = './Fig_roi_mapping/' + self.expdate + '/' + self.cell_id
            plt.savefig(img_save_path + '/{}-{}-adjusted.png'.format(self.expdate,dname))

        #############

        if not adjust:
    #         print('If this looks right, set `adjust` to True to adjust the ROI coordinates.')
            print('')
        else:
            print('The ROI coordinates have been adjusted!')
            d_coords_xy = np.round(roi_coords_stack_xy).astype(int)

            dict_coords = {}
            dict_rec_center = {}
            # print(idx_subframe)
            for iii, idx in enumerate(idx_subframe):
                # print(iii, idx, len(d_coords_xy))
                dict_coords[idx] = d_coords_xy[iii]
                dict_rec_center[idx] = np.repeat(rec_center_stack_xy.reshape(1,2), len(idx_subframe), axis=0)[iii]

            self.df_rois.set_value(idx_subframe, 'recording_center', dict_rec_center)
            self.df_rois.set_value(idx_subframe, 'roi_coords_stack_xy', dict_coords)  


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

            filename = np.unique(df_sub['filename'])[0]

            d = load_h5_data(self.celldir + '/Pre/' + filename)
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
                Tracetimes0_dict[idx] = d['Tracetimes0'][:, roi_id-1]
                Traces0_raw_dict[idx] = d['Traces0_raw'][:, roi_id-1]
                rf_s_dict[idx] = d['STRF_SVD_Space0'][:, :, roi_id-1]
                rf_t_dict[idx] = d['STRF_SVD_Time0'][:, roi_id-1]

        self.df_rois['roi_coords'] = pd.Series(roi_coords_dict)
        self.df_rois['path_id'] = pd.Series(path_id_dict)
        self.df_rois['Triggervalues'] = pd.Series(Triggervalues_dict)
        self.df_rois['Triggertimes'] = pd.Series(Triggertimes_dict)
        self.df_rois['Tracetimes0'] = pd.Series(Tracetimes0_dict)
        self.df_rois['Traces0_raw'] = pd.Series(Traces0_raw_dict)
        self.df_rois['rf_s'] = pd.Series(rf_s_dict)
        self.df_rois['rf_t'] = pd.Series(rf_t_dict)


    def check_all_rois(self, plot_bands=False):
        self._three_views(plot_rois=True, plot_bands=plot_bands)
    # def plot_all_rois(self):

    #     linestack = trace2linestack(self.df_trace, self.meta_trace)

    #     linestack_xy = linestack.sum(2)
    #     linestack_xy[linestack_xy != 0] = 1

    #     plt.figure(figsize=(16,16))
    #     plt.imshow(linestack_xy, origin='lower', cmap=plt.cm.binary)

    #     plt.imshow(self.info_soma['mask_xy'], origin='lower', cmap=plt.cm.binary, vmin=0.0, alpha=0.3)

    #     rois_pos = np.vstack(self.df_rois.roi_coords)
    #     rois_dis = self.df_rois.distance_dendritic.values
    #     plt.scatter(rois_pos[:, 1], rois_pos[:, 0], c=rois_dis/self.df_rois.distance_dendritic.max(), s=180, cmap=plt.cm.viridis)
        
    #     for i, roi_id in enumerate(self.df_rois.index):
    #         plt.annotate(int(roi_id), xy=(rois_pos[i][1]-3, rois_pos[i][0]-3), color='white', fontsize=16*0.7)
    #     scalebar = ScaleBar(self.stack_pixel_size, units='um', location='lower left', box_alpha=0, pad=4)
    #     plt.gca().add_artist(scalebar)

    def update_df_rois_with_segments_and_branchpoints(self):

        df_rois = self.df_rois.copy()
        df_paths = self.df_paths.copy()

        branchpoints_dict = {}
        num_branchpoints_dict = {}
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

            sm = np.vstack([self.info_soma['centroid'], sm])
            
            if bpts == [[]]:
                bpts = np.array([self.info_soma['centroid']]) # soma centroid is always a branchpoint
            else:
                bpts = np.vstack([bpts, self.info_soma['centroid']]) # soma centroid at the last pos
                # bpts = np.vstack([self.info_soma['centroid'] , bpts]) # soma centroid at the first pos

            branchpoints_dict[idx] = bpts
            num_branchpoints_dict[idx] = len(bpts)
            segments_dict[idx] = sm
            distance_dendritic_dict[idx] = np.sum(np.sqrt(np.sum((sm[1:] - sm[:-1])**2, 1))) * self.stack_pixel_size
            distance_radial_dict[idx] = np.sqrt(np.sum((self.info_soma['centroid'] - df_rois.loc[idx].roi_coords) ** 2)) * self.stack_pixel_size
        
        df_rois['branchpoints'] = pd.Series(branchpoints_dict)
        df_rois['num_branchpoints'] = pd.Series(num_branchpoints_dict)
        df_rois['segments'] = pd.Series(segments_dict)
        df_rois['distance_dendritic'] = pd.Series(distance_dendritic_dict)
        df_rois['distance_radial'] = pd.Series(distance_radial_dict)
        
        self.df_rois = df_rois.copy()

    def plot_all_roi2soma(self):

        df_rois = self.df_rois.copy()

        for row in df_rois.iterrows():
            
            idx = row[0]
            rec_id = row[1]['recording_id']
            roi_id = row[1]['roi_id']
            
            linestack = trace2linestack(self.df_trace, self.meta_trace)
            
            plt.figure(figsize=(16,16))
            plt.imshow(linestack.sum(2), origin='lower', cmap=plt.cm.binary) # Cell Skeleton
            plt.imshow(self.info_soma['mask_xy'], origin='lower', cmap=plt.cm.binary, vmin=0.0, alpha=0.3) # Soma
            plt.grid('off')

            all_rois_pos = np.vstack(df_rois.roi_coords) # all rois
            plt.scatter(all_rois_pos[:, 1], all_rois_pos[:, 0], c='gray', s=180, alpha=0.3)

            sm = self.df_rois.loc[idx].segments # dendritic trace from roi to soma
            plt.plot(sm[:, 1], sm[:, 0], lw=2.5, color='red')

            bpts = df_rois.loc[idx].branchpoints
            roi_pos = df_rois.loc[idx].roi_coords

            # if len(bpts.shape) == 1:
            #     plt.scatter(bpts[1], bpts[0], color='black', s=80, zorder=10)
            # else:
            #     plt.scatter(bpts[:, 1], bpts[:, 0], color='black', s=80, zorder=10) # bpts from roi to soma
            plt.scatter(bpts[:, 1], bpts[:, 0], color='black', s=80, zorder=10) # bpts from roi to soma
            plt.scatter(roi_pos[1], roi_pos[0], color='red', s=180) # roi of interest

            scalebar = ScaleBar(self.stack_pixel_size, units='um', location='upper left', box_alpha=0)
            plt.gca().add_artist(scalebar)
            
            plt.title('rec {}: roi {}'.format(rec_id, roi_id))
            
            img_save_path = 'img-roi2soma/' + self.expdate + '/' + self.exp_num
            if not os.path.exists(img_save_path):
                    os.makedirs(img_save_path) 
            
            plt.savefig(img_save_path + '/{}-{}-{}.png'.format(idx, int(rec_id), int(roi_id)))


    #########################
    ## ROI Receptive Field ##
    #########################

    def check_ROI_RFs(self, stimulus_path, rf_pixel_size=30, sigma=0.5, stdev=2, fit_gaussian=False,special_cases=None, thresholded=False, plot_thresholded=False, plot_gaussian_fit=False):

        import sta_utils as stools
        import scipy.ndimage as ndimage
        import cv2

        num_subplots_per_row = np.ceil(self.df_rois.index.size / 4).astype(int)
        fig, ax = plt.subplots(num_subplots_per_row, 4, figsize=(16, 4 * num_subplots_per_row))

        
        stack_pixel_size = self.stack_pixel_size
        RF_scale = np.array([15, 20]) * (rf_pixel_size / stack_pixel_size)
        Rx, Ry = RF_scale[0]/2, RF_scale[1]/2

        df_rois = self.df_rois.copy()
        exception_list = []
        for row in df_rois.iterrows():

            idx = row[0] # the real roi_id for all rois
            roi_id = row[1]['roi_id'] # relative roi_id within each recording
            Sx, Sy = row[1]['recording_center'] 
            left_pad = np.round(Ry - Sy).astype(int)
            bottom_pad = np.round(Rx - Sx).astype(int)

            RF = stools.extract_sta_rois(df_rois, idx, stimulus_path)
            RF = np.fliplr(RF)
            RFcp = RF.copy()
            
            RF = ndimage.gaussian_filter(RF, sigma=(sigma, sigma), order=0)
            RFcp = ndimage.gaussian_filter(RF, sigma=(sigma, sigma), order=0)

            if special_cases != None and idx in special_cases.keys():
                if thresholded:
                    RF[np.logical_and(RF < RF.mean() + RF.std() * 3, RF > RF.mean() - RF.std() * 3)] = 0 
                if fit_gaussian:
                    (cntr_x, cntr_y), RF = stools.get_contour(RF=stools.gaussian_fit(RF), stdev=special_cases[idx]['stdev'])
                    if plot_gaussian_fit:
                        RFcp = RF.copy()
                else:
                    (cntr_x, cntr_y), RF = stools.get_contour(RF=RF, stdev=special_cases[idx]['stdev'])
            else:
                if thresholded:
                    RF[np.logical_and(RF < RF.mean() + RF.std() * 3, RF > RF.mean() - RF.std() * 3)] = 0
                if fit_gaussian:
                    (cntr_x, cntr_y), RF = stools.get_contour(RF=stools.gaussian_fit(RF), stdev=stdev)
                    if plot_gaussian_fit:
                        RFcp = RF.copy()
                else:
                    (cntr_x, cntr_y), RF = stools.get_contour(RF=RF, stdev=stdev)

            RF_size = cv2.contourArea(np.vstack([cntr_x, cntr_y]).T.astype(np.float32)) * 0.9

            if RF_size < 1.5:
                print('ROI {} has RF size smaller than 1.5'.format(idx))
                exception_list.append(idx)
            
            if plot_thresholded:
                ax[np.floor(idx/4).astype(int), (idx % 4).astype(int)].imshow(RF, origin='lower')
            else:
                ax[np.floor(idx/4).astype(int), (idx % 4).astype(int)].imshow(RFcp, origin='lower')
            ax[np.floor(idx/4).astype(int), (idx % 4).astype(int)].axis('off')
            ax[np.floor(idx/4).astype(int), (idx % 4).astype(int)].plot(cntr_y, cntr_x, color='red')
            ax[np.floor(idx/4).astype(int), (idx % 4).astype(int)].set_title('ROI {}  ({} $x 10^3$ $um^2$)'.format(idx, round(RF_size, 2), fontsize=10))

        num_RF = self.df_rois.index.size
        num_subplots = round(self.df_rois.index.size / 4) *  4
        if num_RF != num_subplots:
            for residul in range(1, num_subplots - num_RF+1):
                ax[np.floor(idx/4).astype(int), (idx % 4).astype(int) + residul].axes.get_xaxis().set_visible(False)
                ax[np.floor(idx/4).astype(int), (idx % 4).astype(int) + residul].axes.get_yaxis().set_visible(False)
                ax[np.floor(idx/4).astype(int), (idx % 4).astype(int) + residul].axis('off')

        fig.suptitle('{}'.format(self.expdate), fontsize=28)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])  
        
        self.RF_exception_list = exception_list

    def check_ROI_contours(self, stimulus_path, rf_pixel_size=30, fit_gaussian=False, sigma=0.5, stdev=2, special_cases=None, thresholded=False):

        import sta_utils as stools
        import scipy.ndimage as ndimage
        from matplotlib_scalebar.scalebar import ScaleBar
        import cv2

        
        stack_pixel_size = self.stack_pixel_size
        RF_scale = np.array([15, 20]) * (rf_pixel_size / stack_pixel_size)
        Rx, Ry = RF_scale[0]/2, RF_scale[1]/2

        roi_rf_cntr = {}
        roi_rf_size = {}
        df_rois = self.df_rois.copy()

        color_arg = np.floor((df_rois.distance_dendritic / 300) * 255)
        color_palletes = np.array(plt.cm.viridis.colors)

        linestack_xy = self.linestack.mean(2)

        linestack_xy[linestack_xy != 0] = 1
        linestack_xy = np.pad(linestack_xy, ((0,55), (0,0)), 'constant')

        plt.figure(figsize=(16,16))
        plt.imshow(linestack_xy, origin='lower')
        plt.imshow(self.info_soma['mask_xy'], origin='lower', cmap=plt.cm.binary, vmin=0.0, alpha=0.3)

        rois_pos = np.vstack(df_rois.roi_coords)
        rois_pos = np.delete(rois_pos, self.RF_exception_list, axis=0)

        rois_dis = df_rois.distance_dendritic.values
        if self.RF_exception_list:
            rois_dis = np.delete(rois_dis, self.RF_exception_list)

        sc = plt.scatter(rois_pos[:, 1], rois_pos[:, 0], c=rois_dis, s=180, cmap=plt.cm.viridis, vmin=0, vmax=300)

        scalebar = ScaleBar(self.stack_pixel_size, units='um', location='lower left', box_alpha=0, pad=4)
        plt.gca().add_artist(scalebar)
        plt.autoscale(True)


        roi_rf_cntr = {}
        roi_rf_size = {}
        roi_rf_offset = {}
        for row in df_rois.iterrows():

            idx = row[0]

            if idx in self.RF_exception_list:

                roi_rf_size[idx] = np.nan
                roi_rf_cntr[idx] = np.nan
                roi_rf_offset[idx] = np.nan

                continue

            roi_id = row[1]['roi_id']
            Sx, Sy = row[1]['recording_center']
            left_pad = np.round(Ry - Sy).astype(int)
            bottom_pad = np.round(Rx - Sx).astype(int)

            RF = stools.extract_sta_rois(df_rois, idx, stimulus_path)
            RF = np.fliplr(RF)
            RFcp = RF.copy()

            RF = ndimage.gaussian_filter(RF, sigma=(sigma, sigma), order=0)
            RFcp = ndimage.gaussian_filter(RF, sigma=(sigma, sigma), order=0)

            if special_cases != None and idx in special_cases.keys():
                if thresholded:
                    RF[np.logical_and(RF < RF.mean() + RF.std() * 3, RF > RF.mean() - RF.std() * 3)] = 0 
                if fit_gaussian:
                    (cntr_x, cntr_y), RF = stools.get_contour(RF=stools.gaussian_fit(RF), stdev=special_cases[idx]['stdev'])
                else:
                    (cntr_x, cntr_y), RF = stools.get_contour(RF=RF, stdev=special_cases[idx]['stdev'])
            else:
                if thresholded:
                    RF[np.logical_and(RF < RF.mean() + RF.std() * 3, RF > RF.mean() - RF.std() * 3)] = 0 
                if fit_gaussian:
                    (cntr_x, cntr_y), RF = stools.get_contour(RF=stools.gaussian_fit(RF), stdev=stdev)
                else:
                    (cntr_x, cntr_y), RF = stools.get_contour(RF=RF, stdev=stdev)

            cntr_x *= rf_pixel_size/self.stack_pixel_size
            cntr_y *= rf_pixel_size/self.stack_pixel_size   

            RF_size = cv2.contourArea(np.vstack([cntr_x, cntr_y]).T.astype(np.float32)) * (self.stack_pixel_size * self.stack_pixel_size)/1000
            if RF_size > 1.5:
                plt.plot(cntr_y-left_pad, cntr_x-bottom_pad, color=color_palletes[color_arg[idx].astype(int)])
            else:
                print(idx, RF_size)
            cntr = np.vstack([cntr_x-bottom_pad, cntr_y-left_pad]).T
            roi_pos = row[1]['roi_coords']
            cntr_pos = cntr.mean(0)
            doffset = cntr_pos - roi_pos[:2]

            roi_rf_size[idx] = RF_size     
            roi_rf_cntr[idx] = cntr
            roi_rf_offset[idx] = doffset

            plt.axis('off')

        cbar = plt.colorbar(sc, fraction=0.02, pad=-.001 )
        cbar.outline.set_visible(False)

        df_rois['RF_size'] = pd.Series(roi_rf_size)
        df_rois['RF_contour'] = pd.Series(roi_rf_cntr)
        df_rois['RF_offset'] = pd.Series(roi_rf_offset)

        self.df_rois = df_rois.copy()

    def check_soma_contour(self, stimulus_path, sigma=0.5, stdev=2, show_figure=False):

        import sta_utils as stools
        import scipy.ndimage as ndimage
        from matplotlib_scalebar.scalebar import ScaleBar
        import cv2

        rf_pixel_size = 30
        stack_pixel_size = self.stack_pixel_size
        RF_scale = np.array([15, 20]) * (rf_pixel_size / stack_pixel_size)
        Rx, Ry = RF_scale[0]/2, RF_scale[1]/2
        Sx, Sy = self.info_soma['centroid'][:2]
        left_pad = np.round(Ry - Sy).astype(int)
        bottom_pad = np.round(Rx - Sx).astype(int)
        
        RF_soma = stools.extract_sta_soma(soma_data=self.data_soma.copy(), stimulus_path=stimulus_path)
        RF_soma = np.fliplr(RF_soma)
        RF_soma = ndimage.gaussian_filter(RF_soma, sigma=(sigma, sigma), order=0)
        
        RF_soma[np.logical_and(RF_soma < RF_soma.mean() + RF_soma.std() * 3, RF_soma > RF_soma.mean() - RF_soma.std() * 3)] = 0

        (cntr_x, cntr_y), RF_soma = stools.get_contour(RF=RF_soma, stdev=stdev)

        if show_figure:
            plt.imshow(RF_soma, origin='lower')
            plt.plot(cntr_y, cntr_x, c='white')
        
        cntr_x *= rf_pixel_size/stack_pixel_size
        cntr_y *= rf_pixel_size/stack_pixel_size
        
        cntr_soma = np.vstack([cntr_x-bottom_pad, cntr_y-left_pad]).T
        
        cntr_soma_center = cntr_soma.mean(0) # RF center

        linestack_xy = self.linestack.mean(2)
        linestack_xy[linestack_xy != 0] = 1
        linestack_xy = ndimage.gaussian_filter(linestack_xy, sigma=30, order=0)
        DF_center = np.array(ndimage.measurements.center_of_mass(linestack_xy)) # DF center
        # stack_soma_center = np.array([Sx, Sy])  
        # offset = cntr_soma_center - stack_soma_center
        offset = cntr_soma_center - DF_center
        self.info_soma.update({'cntr_soma': cntr_soma,
                                'offset': offset,
                                'DF_center': DF_center})

        with open(self.IMT_Data_path + '/info_soma_updated.pickle', 'wb') as output_file:
            pickle.dump(self.info_soma, output_file)        

    ###############
    ## ROI Pairs ##
    ###############

    def get_roipairs(self):

        df_roipairs = pd.DataFrame(columns=('pair_id',
                                        'average_radial_distance_to_soma',
                                        'average_dendritic_distance_to_soma',
                                        'segments_between',
                                        'branchpoints_between',
                                        'dendritic_distance_between',
                                        'radial_distance_between',
                                        'angle_between',
                                        'num_branchpoints_between',
                                        'distance_between_RFs_center',
                                        'overlap_contour',
                                        'overlap_RFsize',
                                        'smaller_RFsize',
                                        'overlap_index'))

        indices = np.array(self.df_rois.index)

        print('{} pairs of ROIs are being processed.'.format(np.sum(np.arange(len(indices)))))

        i = 0
        for roi_id0 in indices:

            indices = np.delete(indices, 0) # delete the first values in every loop

            for roi_id1 in indices:
                

                print('Processing pair ({} {})...'.format(roi_id0, roi_id1))
                
                ''' START: pair dendritic distance and branchpoints '''
                
                sms_between, bpts_between = get_info_roi2roi(self.df_trace, self.df_paths, self.df_rois, self.info_soma, roi_id0, roi_id1)
                
                dendritic_distance_between = 0
                for sm in sms_between:
                    dendritic_distance_between += np.sum(np.sqrt(np.sum((sm[1:] - sm[:-1])**2, 1))) * self.stack_pixel_size

                ''' END: pair dendritic distance and branchpoints '''

                
                ''' START: pair angle '''
                
                angle_deg_between = get_angle_roi2roi(self.df_rois, self.info_soma, roi_id0, roi_id1, dim=2)
                
                ''' END: pair angle'''

                ''' START: radial distance between two rois '''
                
                radial_distance_between = np.sqrt(np.sum((self.df_rois.loc[roi_id0].roi_coords - self.df_rois.loc[roi_id1].roi_coords) ** 2)) * self.stack_pixel_size

                ''' END: radial distance between two rois '''


                ''' START: contours overlap  '''
                cntr0 = self.df_rois.loc[roi_id0].RF_contour
                cntr1 = self.df_rois.loc[roi_id1].RF_contour

                if np.isnan(cntr0).any() or np.isnan(cntr1).any():
                    
                    overlap_RFsize = 0
                    overlap_index = 0

                    if np.isnan(cntr0).any():
                        smaller_RFsize = self.df_rois.loc[roi_id1].RF_size
                    elif np.isnan(cntr1).any():
                        smaller_RFsize = self.df_rois.loc[roi_id0].RF_size
                    else:
                        smaller_RFsize = np.nan

                else: 

                    num_intp = 100
                    overlap_cntr, RFsizes = get_inner_cntr(cntr0, cntr1, num_intp, self.stack_pixel_size)
                    overlap_RFsize = RFsizes['overlap_size']
                    smaller_RFsize = RFsizes['smaller_size']
                    overlap_index = overlap_RFsize/smaller_RFsize

                    cntr0_center = cntr0.mean(0)
                    cntr1_center = cntr1.mean(0)

                    distance_between_RFs_center = np.sqrt(np.sum((cntr0_center - cntr1_center) ** 2)) * self.stack_pixel_size

                ''' END: contours overlap  '''

                
                '''START: average radial distance'''

                radial_distance_roi0 = self.df_rois.loc[roi_id0].distance_radial
                radial_distance_roi1 = self.df_rois.loc[roi_id1].distance_radial
                average_radial_distance = np.mean(radial_distance_roi0 + radial_distance_roi1)

                '''END: average radial distance'''

                
                '''START: average dendritic distance'''

                dendritic_distance_roi0 = self.df_rois.loc[roi_id0].distance_dendritic
                dendritic_distance_roi1 = self.df_rois.loc[roi_id1].distance_dendritic
                average_dendritic_distance = np.mean(dendritic_distance_roi0 + dendritic_distance_roi1)
                
                '''END: average dendritic distance'''

                df_roipairs.loc[i] = [set([roi_id0, roi_id1]),      # pair_id
                                        average_radial_distance,    # average_radial_distance_to_soma
                                        average_dendritic_distance, # average_dendritic_distance_to_soma
                                        sms_between,                # segments_between
                                        bpts_between,               # branchpoints_between
                                        dendritic_distance_between, # dendritic_distance_between
                                        radial_distance_between,    # radial_distance_between
                                        angle_deg_between,          # angle_between
                                        len(bpts_between),          # num_branchpoints_between
                                        distance_between_RFs_center,# distance_between_RFs_center
                                        overlap_cntr,               # overlap_contour
                                        overlap_RFsize,             # overlap_RFsize
                                        smaller_RFsize,             # smaller_RFsize
                                        overlap_index]              # overlap_index
                i+= 1

        print('Done!')

        self.df_roipairs = df_roipairs

    def plot_all_roipairs(self):

        indices = np.array(self.df_rois.index)
        linestack = trace2linestack(self.df_trace, self.meta_trace)
        idx = 0
        for roi_id0 in indices:

            indices = np.delete(indices, 0) # delete the first values in every loop

            for roi_id1 in indices:
                
                plt.figure(figsize=(16,16))
                plt.imshow(linestack.sum(2), origin='lower', cmap=plt.cm.binary)
                plt.imshow(self.info_soma['mask_xy'], origin='lower', cmap=plt.cm.binary, vmin=0.0, alpha=0.3)

                all_roi_pos = np.vstack(self.df_rois.roi_coords)
                plt.scatter(all_roi_pos[:, 1], all_roi_pos[:, 0], color='gray', s=180, alpha=0.3)

                bpts = self.df_roipairs[self.df_roipairs.pair_id == {roi_id0,roi_id1}].branchpoints_between.values[0]
                sms = self.df_roipairs[self.df_roipairs.pair_id == {roi_id0,roi_id1}].segments_between.values[0]

                roi0_pos = self.df_rois.loc[roi_id0].roi_coords
                roi1_pos = self.df_rois.loc[roi_id1].roi_coords

                plt.scatter(roi0_pos[1], roi0_pos[0], s=180, color='red')
                plt.scatter(roi1_pos[1], roi1_pos[0], s=180, color='red')

                for bpt in bpts:
                    if len(bpt) != 0:
                        plt.scatter(bpt[1], bpt[0], s=80, color='black', zorder=10)

                for sm in sms:
                    plt.plot(sm[:, 1], sm[:, 0], color='red', lw=4)

                scalebar = ScaleBar(self.stack_pixel_size, units='um', location='upper left', box_alpha=0)
                plt.gca().add_artist(scalebar)

                img_save_path = 'img-roi2roi/' + self.expdate + '/' + self.exp_num
                if not os.path.exists(img_save_path):
                        os.makedirs(img_save_path) 
                
                plt.savefig(img_save_path + '/{}_{}to{}.png'.format(idx, int(roi_id0), int(roi_id1)))

                idx += 1

    def plot_RF_offset(self):

        for roi_idd in range(0, t.df_rois.index.max()+1):
            roi_loc = t.df_rois.loc[roi_idd].roi_coords
            RF_loc  = t.df_rois.loc[roi_idd].RF_contour.mean(0)
            doffset = RF_loc - roi_loc[:2]
        #     plt.plot(t.df_rois.loc[roi_idd].RF_contour[:, 1], t.df_rois.loc[roi_idd].RF_contour[:, 0])
        #     plt.scatter(RF_loc[1], RF_loc[0], s=280, color='red')
        #     plt.scatter(roi_loc[1], roi_loc[0], s=280, color='orange')
            plt.arrow(roi_loc[1], roi_loc[0], doffset[1], doffset[0], fc='k', ec='k', width=1, head_width=5, head_length=5)

    def get_df_branchpoints(self):

        self.df_branchpoints = get_df_branchpoints(self.df_trace, self.df_rois, self.df_paths, self.stack_pixel_size, self.info_soma)


    ####################
    ## Misc: Plotting ##
    ####################

    def plot_contours(self, manually_adjust=np.array([0,0]), adjust=False, plot_arrows=False):

        df_rois = self.df_rois.copy()

        color_arg = np.floor((df_rois.distance_dendritic / 300) * 255)
        color_palletes = np.array(plt.cm.viridis.colors)

        plt.figure(figsize=(16,16))
        plt.imshow(self.linestack.sum(2), origin='lower', cmap=plt.cm.binary)
        plt.imshow(self.info_soma['mask_xy'], origin='lower', cmap=plt.cm.binary, vmin=0.0, alpha=0.3)
        
        rois_pos = np.vstack(df_rois.roi_coords)
        rois_dis = df_rois.distance_dendritic.values
        
        sc = plt.scatter(rois_pos[:, 1], rois_pos[:, 0], c=rois_dis, s=180, cmap=plt.cm.viridis, vmin=0, vmax=300)
        
        scalebar = ScaleBar(self.stack_pixel_size, units='um', location='lower left', box_alpha=0, pad=4)
        plt.gca().add_artist(scalebar)
        
        if adjust:
            offset = self.info_soma['offset'] + manually_adjust
        
        for row in df_rois.iterrows():

            idx = row[0]

            if idx in self.RF_exception_list:
                continue


            roi_id = row[1]['roi_id']
            cntr = row[1]['RF_contour'] # cntr_original

            if adjust:
                cntr = cntr - offset  # cntr_adjusted
            
            cntr_x = cntr[:, 0]
            cntr_y = cntr[:, 1]
             
        #     plt.plot(cntr_y, cntr_x, color='gray')    
            plt.plot(cntr_y, cntr_x, color=color_palletes[color_arg[idx].astype(int)])

            if plot_arrows:
                roi_pos = row[1]['roi_coords']
                cntr_pos = cntr.mean(0)
                doffset = cntr_pos - roi_pos[:2]
                plt.arrow(roi_pos[1], roi_pos[0], doffset[1], doffset[0], fc='k', ec='k', width=1, head_width=5, head_length=5, zorder=99)

        plt.autoscale(True)
        plt.axis('off')


    def plot_trend_distance_vs_RFsize(self):
        
        row_to_delete = self.RF_exception_list
        def delete_row(arr):
            return np.delete(arr, row_to_delete)
        
        x0 = delete_row(self.df_rois.distance_radial.as_matrix())
        x1 = delete_row(self.df_rois.distance_dendritic.as_matrix())
        y  = delete_row(self.df_rois.RF_size.as_matrix())
        
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].scatter(x0, y, color='black')
        ax[0].set_xlabel('Radial distance to soma')
        ax[0].set_ylabel('RF size')
        ax[0].set_xlim(0, 270)
        
        ax[1].scatter(x1, y, color='black')
        ax[1].set_xlabel('Dendritic distance to soma')
        ax[1].set_ylabel('RF size')
        ax[1].set_xlim(0, 270)