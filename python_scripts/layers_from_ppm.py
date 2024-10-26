import os
import sys
import json
import re
import math
import numpy as np
import cv2
import tifffile
from scipy.interpolate import RegularGridInterpolator
import psutil

class LayersFromPPM():
    """A python implementation of Volume Cartographer's layers_from_ppm utility."""

    def __init__(self):
        self.volpkg_dir = ''
        self.ppm_filepath = ''
        self.volume_name = ''
        self.transform_filepath = ''
        self.invert_transform = False
        self.radius = None
        self.sampling_interval = 1.0
        self.sampling_direction = 0
        self.output_dir = ''
        self.cache_mem_limit = None
        self.image_format = 'png'
        self.compression = 1
        self.dtype = np.float64
        self.progress_handler = None
        
        
    def run(self):

        # Get volume info
        with open(os.path.join(self.volpkg_dir, 'volumes', self.volume_name, 'meta.json')) as f:
            meta_info = json.load(f)
            vd, vh, vw =  meta_info['slices'], meta_info['height'], meta_info['width']
            vshape = (vd, vh, vw)
            volimg_name_len = len(str(vd-1))
            voxel_size = meta_info['voxelsize']

        # Setup the sampling range
        if self.radius is None:
            with open(os.path.join(self.volpkg_dir, 'config.json')) as f:
                self.radius = json.load(f)['materialthickness'] / (2 * voxel_size)
        rMin = 0 if self.sampling_direction == 1 else -self.radius
        rMax = 0 if self.sampling_direction == -1 else self.radius
        num_layers = math.floor((rMax - rMin)/self.sampling_interval) + 1
    
        # Get the transformation matrix
        xfrm = np.identity(4)
        if self.transform_filepath:
            with open(self.transform_filepath) as f:
                xfrm = np.array(json.load(f)['params'])
                if self.invert_transform:
                    xfrm = np.linalg.inv(xfrm)
            xfrm = xfrm.astype(self.dtype)
    
        # Read the ppm file
        iw, ih, sample_pts, normals, ppm_mask = self.read_ppm_file()
    
        # Get the extents of the layers volume
        samples_ext = np.concatenate(((sample_pts + rMin*normals)[ppm_mask,:], (sample_pts + rMax*normals)[ppm_mask,:]))
        samples_ext = np.matmul(xfrm, np.c_[samples_ext, np.ones(samples_ext.shape[0], dtype=samples_ext.dtype)].T).T[:, :3]
        vx_min, vy_min, vz_min = (max(math.floor(np.amin(samples_ext[:, i])) - 1, 0) for i in range(3))
        vx_max, vy_max, vz_max = (min(math.ceil (np.amax(samples_ext[:, i])) + 1, vshape[2-i]) for i in range(3))
        del samples_ext
        sample_pts = sample_pts.reshape((ih, iw, 3))
        normals = normals.reshape((ih, iw, 3))
        ppm_mask = ppm_mask.reshape((ih, iw))

        # Determine buffer sizes based on image sizes and available memory
        crop_h = vy_max - vy_min + 1
        crop_w = vx_max - vx_min + 1
        linsp_x = np.linspace(vx_min, vx_max, crop_w)
        linsp_y = np.linspace(vy_min, vy_max, crop_h)
    
        mem_budget = 0.5 * psutil.virtual_memory().total if self.cache_mem_limit is None else self.cache_mem_limit
        num_xchunks = round(max(1, 224*iw*ih/(mem_budget/3))) # Account for mem needed by sample_pts, normals, and the Interpolator
        xchunk_size = min(math.ceil(iw/num_xchunks), iw)
        avail_mem = max(1, mem_budget - 224*ih*xchunk_size)
        num_layer_batches = max(1, min(num_layers, round(num_layers*(ih*iw*2)/(0.5*avail_mem))) )
        layers_per_batch = round(num_layers/num_layer_batches)
        avail_mem -= layers_per_batch * (ih*iw*2)
        subvol_depth = max(8, min(vz_max-vz_min+1, int(avail_mem/(4*crop_h*crop_w))))
    
        # Get info needed for progress reporting
        num_subvolumes = 1 if subvol_depth == vz_max-vz_min+1 else math.ceil((vz_max-vz_min+1)/(subvol_depth-1))
        num_imreads_needed = num_layer_batches * ((vz_max - vz_min + 1) + (num_subvolumes - 1))
        num_interps_needed = num_layers * num_subvolumes
        total_ops = num_imreads_needed/50 + num_interps_needed
        num_imreads_done = 0
        num_interps_done = 0
        if self.progress_handler:
            self.progress_handler(0, total_ops)

        # Allocate working memory
        layer_imgs = np.empty((layers_per_batch, ih, iw), dtype=np.uint16)
        subvol = np.zeros((subvol_depth, crop_h, crop_w), dtype=np.float32)
    
        # Iterate over layer batches
        os.makedirs(self.output_dir, exist_ok=True)
        for layer_batch_start in range(0, num_layers, layers_per_batch):
            layer_batch_end = min(num_layers, layer_batch_start + layers_per_batch)
            layer_imgs[:, :, :] = 0
    
            # Iterate over sub-volumes (with a 1-slice overlap)
            vz_step = subvol_depth if subvol_depth == vz_max-vz_min+1 else subvol_depth - 1
            for vz_chunk_start in range(vz_min, vz_max+1, vz_step):
                vz_chunk_end = min(vz_max+1, vz_chunk_start+subvol_depth)
    
                # Load the sub-volume
                for vz in range(vz_chunk_start, vz_chunk_end):
                    img_path = os.path.join(self.volpkg_dir, 'volumes', self.volume_name, str(vz).zfill(volimg_name_len) + '.tif')
                    subvol[vz - vz_chunk_start, :, :] = tifffile.imread(img_path)[vy_min:vy_max + 1, vx_min:vx_max + 1]
    
                    # Report progress
                    num_imreads_done += 1
                    if self.progress_handler and (num_imreads_done%50) == 0:
                        self.progress_handler(num_imreads_done/50 + num_interps_done, total_ops)

                # Interpolate. (Image pixels that don't intersect the current sub-volume will just be
                # set to zero, and over-written in a later iteration.)
                linsp_z = np.linspace(vz_chunk_start, vz_chunk_end-1, vz_chunk_end - vz_chunk_start)
                interp = RegularGridInterpolator((linsp_z, linsp_y, linsp_x),
                                                 subvol[0:vz_chunk_end - vz_chunk_start, :, :],
                                                 method='linear', bounds_error=False, fill_value=0)


                for layer_id in range(layer_batch_start, layer_batch_end):
                    layer_img = layer_imgs[layer_id - layer_batch_start]

                    for ix_chunk_start in range(0, iw, xchunk_size):
                        ix_chunk_end = min(iw, ix_chunk_start + xchunk_size)
                        ix_slice = np.index_exp[:, ix_chunk_start:ix_chunk_end]
                        
                        pts = sample_pts[ix_slice] + (rMin + layer_id*self.sampling_interval) * normals[ix_slice]
                        pts = np.concatenate((pts, np.ones(pts.shape[:2] + (1,), dtype=pts.dtype)), axis=2)
                        pts = np.matmul(xfrm, pts.reshape((-1,4)).T)
                        pts = pts.T[:, :3][:, [2, 1, 0]].reshape(sample_pts[ix_slice].shape)
                        pts[~ppm_mask[ix_slice], :] = [-1, -1, -1]

                        interp_vals = np.rint(interp(pts)).astype(np.uint16)
                        np.maximum(layer_img[ix_slice], interp_vals, out=layer_img[ix_slice])

                    # Report progress
                    num_interps_done += 1
                    if self.progress_handler and (abs(num_interps_done%1) < 1e-4):
                        self.progress_handler(num_imreads_done/50 + num_interps_done, total_ops)

                    # Save the result if done
                    if vz_chunk_start + vz_step >= vz_max:
                        fname = str(layer_id).zfill(len(str(num_layers - 1))) + \
                                ('.tif' if self.image_format.lower().startswith('tif') else '.png')
                        params = [int(cv2.IMWRITE_PNG_COMPRESSION), max(0, min(9, self.compression))] if \
                            fname.endswith('png') else []
                        cv2.imwrite(os.path.join(self.output_dir, fname), layer_img, params)
    

    def read_ppm_file(self):
        with open(self.ppm_filepath, 'rb') as ppm_file:
            # Read the header
            header = {}
            while True:
                line = ppm_file.readline().decode('utf-8').rstrip("\n")
                if line.startswith('<>'):
                    break
                key, val = line.split(': ', 1)
                header[key] = int(val) if val.isnumeric() else val
            iw = header['width']
            ih = header['height']

            # Read the data in chunks
            sample_pts = np.zeros((ih*iw, 3), dtype=self.dtype)
            normals = np.zeros((ih*iw, 3), dtype=self.dtype)
            y_step = max(1, int(2**29/(iw*6*sample_pts.itemsize))) # 512 MB chunks
            for y in range(0, ih, y_step):
                offset = y * iw
                numpxls = iw * min(y_step, ih - y)
                vals = np.fromfile(ppm_file, dtype=np.float64, count=6*numpxls).reshape((-1, 3))
                sample_pts[offset: offset + numpxls, :] = vals[0::2]
                normals[offset: offset + numpxls, :] = vals[1::2]

        # Create the mask. (False for locations with a zero-length normal, True elsewhere)
        mask = np.linalg.norm(normals, axis=1) > 1e-4

        return iw, ih, sample_pts, normals, mask


    @staticmethod
    def default_progress_handler(current, total):
        symbol = '#'
        length = 30
        percent = f'{(100 * current / total):.1f}'
        filled_length = round(length * current / total)
        bar = symbol * filled_length + '-' * (length - filled_length)
        sys.stdout.write(f'\rProgress: |{bar}| {percent}% complete')
        sys.stdout.flush()


    @staticmethod
    def parse_memsize_string(s):
        """
        Interprets a memory-size string of the form '32GB', '256M', etc.
        """
        s = s.strip()
        num = int(re.search(r"^[0-9]+", s).group())
        suffix = re.search(r"([KMGT])?B?$", s).group().upper()

        if suffix[0] == "T":
            return num * (1024 ** 4)
        elif suffix[0] == "G":
            return num * (1024 ** 3)
        elif suffix[0] == "M":
            return num * (1024 ** 2)
        elif suffix[0] == "K":
            return num * 1024
        else:
            return num

