import torch
import torch.nn
import torch.nn.functional as F
import torch.utils.data
import numpy as np
from tqdm import tqdm
import os
import glob

from data.dataset import PairSampleDataset
from models.imitator_temporal_smoothing import Imitator
from options.test_options import TestOptions
from utils.visdom_visualizer import VisdomVisualizer
from utils.util import load_pickle_file, write_pickle_file, mkdirs, mkdir, clear_dir, morph
import utils.cv_utils as cv_utils


__all__ = ['write_pair_info', 'scan_tgt_paths', 'meta_imitate',
           'MetaCycleDataSet', 'make_dataset', 'adaptive_personalize']


@torch.no_grad()
def write_pair_info(src_info, tsf1_info, tsf3_info, out_file, imitator, only_vis):
    """
    Args:
        src_info:
        tsf_info:
        out_file:
        imitator:
    Returns:

    """
    pair_data = dict()

    pair_data['from_face_index_map'] = src_info['fim'][0][:, :, None].cpu().numpy()
    pair_data['to_face_index_map1'] = tsf1_info['fim'][0][:, :, None].cpu().numpy()
    pair_data['to_face_index_map3'] = tsf3_info['fim'][0][:, :, None].cpu().numpy()
    pair_data['T1'] = tsf1_info['T'][0].cpu().numpy()
    pair_data['T3'] = tsf3_info['T'][0].cpu().numpy()
    pair_data['warp1'] = tsf1_info['tsf_img'][0].cpu().numpy()
    pair_data['warp3'] = tsf3_info['tsf_img'][0].cpu().numpy()
    pair_data['smpls1'] = torch.cat([src_info['theta'], tsf1_info['theta']], dim=0).cpu().numpy()
    pair_data['smpls3'] = torch.cat([src_info['theta'], tsf3_info['theta']], dim=0).cpu().numpy()
    pair_data['j2d1'] = torch.cat([src_info['j2d'], tsf1_info['j2d']], dim=0).cpu().numpy()
    pair_data['j2d3'] = torch.cat([src_info['j2d'], tsf3_info['j2d']], dim=0).cpu().numpy()

    tsf1_f2verts, tsf1_fim, tsf1_wim = imitator.render.render_fim_wim(tsf1_info['cam'], tsf1_info['verts'])
    tsf1_p2verts = tsf1_f2verts[:, :, :, 0:2]
    tsf1_p2verts[:, :, :, 1] *= -1

    T_cycle = imitator.render.cal_bc_transform(tsf1_p2verts, src_info['fim'], src_info['wim'])
    pair_data['T1_cycle'] = T_cycle[0].cpu().numpy()

    tsf3_f2verts, tsf3_fim, tsf3_wim = imitator.render.render_fim_wim(tsf3_info['cam'], tsf3_info['verts'])
    tsf3_p2verts = tsf3_f2verts[:, :, :, 0:2]
    tsf3_p2verts[:, :, :, 1] *= -1

    T_cycle = imitator.render.cal_bc_transform(tsf3_p2verts, src_info['fim'], src_info['wim'])
    pair_data['T3_cycle'] = T_cycle[0].cpu().numpy()

    # back_face_ids = mesh.get_part_face_ids(part_type='head_back')
    # tsf_p2verts[:, back_face_ids] = -2
    # T_cycle_vis = imitator.render.cal_bc_transform(tsf_p2verts, src_info['fim'], src_info['wim'])
    # pair_data['T_cycle_vis'] = T_cycle_vis[0].cpu().numpy()

    # for key, val in pair_data.items():
    #     print(key, val.shape)

    write_pickle_file(out_file, pair_data)


def scan_tgt_paths(tgt_path, itv=20, start=0):
    if os.path.isdir(tgt_path):
        all_tgt_paths = glob.glob(os.path.join(tgt_path, '*'))
        all_tgt_paths.sort()
        all_tgt_paths = all_tgt_paths[start:start - 2 if start - 2 != 0 else len(all_tgt_paths):itv]
    else:
        all_tgt_paths = [tgt_path]

    return all_tgt_paths


def meta_imitate(opt, imitator, prior_tgt_path, save_imgs=True, visualizer=None):
    src_path = opt.src_path

    all_tgt_paths1 = scan_tgt_paths(prior_tgt_path, itv=40, start=0)
    all_tgt_paths2 = scan_tgt_paths(prior_tgt_path, itv=40, start=1)
    all_tgt_paths3 = scan_tgt_paths(prior_tgt_path, itv=40, start=2)
    output_dir = opt.output_dir

    out_img_dir, out_pair_dir = mkdirs([os.path.join(output_dir, 'imgs'), os.path.join(output_dir, 'pairs')])

    img_pair_list = []

    for t in tqdm(range(len(all_tgt_paths1))):
        tgt_path1 = all_tgt_paths1[t]
        tgt_path2 = all_tgt_paths2[t]
        tgt_path3 = all_tgt_paths3[t]
        preds = imitator.inference([(tgt_path1, tgt_path2, tgt_path3)], visualizer=visualizer, cam_strategy=opt.cam_strategy, verbose=False)

        tgt_name = os.path.split(tgt_path2)[-1]
        out_path = os.path.join(out_img_dir, 'pred_' + tgt_name)

        if save_imgs:
            cv_utils.save_cv2_img(preds[0], out_path, normalize=True)
            write_pair_info(imitator.src_info, imitator.tsf1_info, imitator.tsf3_info,
                            os.path.join(out_pair_dir, '{:0>8}.pkl'.format(t)), imitator=imitator,
                            only_vis=opt.only_vis)

            img_pair_list.append((src_path, tgt_path1, tgt_path2, tgt_path3))

    if save_imgs:
        write_pickle_file(os.path.join(output_dir, 'pairs_meta.pkl'), img_pair_list)


class MetaCycleDataSet(PairSampleDataset):
    def __init__(self, opt):
        super(MetaCycleDataSet, self).__init__(opt, True)
        self._name = 'MetaCycleDataSet'

    def _read_dataset_paths(self):
        # read pair list
        self._dataset_size = 0
        self._read_samples_info(None, self._opt.pkl_dir, self._opt.pair_ids_filepath)

    def _read_samples_info(self, im_dir, pkl_dir, pair_ids_filepath):
        """
        Args:
            im_dir:
            pkl_dir:
            pair_ids_filepath:

        Returns:

        """
        # 1. load image pair list
        self.im_pair_list = load_pickle_file(pair_ids_filepath)

        # 2. load pkl file paths
        self.all_pkl_paths = sorted(glob.glob((os.path.join(pkl_dir, '*.pkl'))))

        assert len(self.im_pair_list) == len(self.all_pkl_paths), '{} != {}'.format(
            len(self.im_pair_list), len(self.all_pkl_paths)
        )
        self._dataset_size = len(self.im_pair_list)

    def __getitem__(self, item):
        """
        Args:
            item (int):  index of self._dataset_size

        Returns:
            new_sample (dict): items contain
                --src_inputs (torch.FloatTensor): (3+3, h, w)
                --tsf_inputs (torch.FloatTensor): (3+3, h, w)
                --T (torch.FloatTensor): (h, w, 2)
                --head_bbox (torch.IntTensor): (4), hear 4 = [lt_x, lt_y, rt_x, rt_y]
                --valid_bbox (torch.FloatTensor): (1), 1.0 valid and 0.0 invalid.
                --images (torch.FloatTensor): (2, 3, h, w)
                --pseudo_masks (torch.FloatTensor) : (2, 1, h, w)
                --bg_inputs (torch.FloatTensor): (3+1, h, w) or (2, 3+1, h, w) if self.is_both is True
        """
        im_pairs = self.im_pair_list[item]
        pkl_path = self.all_pkl_paths[item]

        sample = self.load_sample(im_pairs, pkl_path)
        sample = self.preprocess(sample)

        sample['preds'] = torch.tensor(self.load_init_preds(im_pairs[2])).float()

        return sample

    def load_init_preds(self, pred_path):
        pred_img_name = os.path.split(pred_path)[-1]
        pred_img_path = os.path.join(self._opt.preds_img_folder, 'pred_' + pred_img_name)
        img = cv_utils.read_cv2_img(pred_img_path)
        img = cv_utils.transform_img(img, self._opt.image_size, transpose=True)
        img = img * 2 - 1

        return img

    def load_sample(self, im_pairs, pkl_path):
        # 1. load images
        imgs = self.load_images(im_pairs)
        # 2.load pickle data
        pkl_data = load_pickle_file(pkl_path)
        src_fim = pkl_data['from_face_index_map'][:, :, 0]  # (img_size, img_size)
        dst1_fim = pkl_data['to_face_index_map1'][:, :, 0]  # (img_size, img_size)
        dst3_fim = pkl_data['to_face_index_map3'][:, :, 0]  # (img_size, img_size)
        T1 = pkl_data['T1']  # (img_size, img_size, 2)
        T3 = pkl_data['T3']  # (img_size, img_size, 2)
        fims1 = np.stack([src_fim, dst1_fim], axis=0)
        fims3 = np.stack([src_fim, dst3_fim], axis=0)

        fims1_enc = self.map_fn[fims1]  # (2, h, w, c)
        fims1_enc = np.transpose(fims1_enc, axes=(0, 3, 1, 2))  # (2, c, h, w)

        fims3_enc = self.map_fn[fims3]  # (2, h, w, c)
        fims3_enc = np.transpose(fims3_enc, axes=(0, 3, 1, 2))  # (2, c, h, w)

        sample = {
            'images': torch.tensor(imgs).float(),
            'src_fim': torch.tensor(src_fim).float(),
            'tsf1_fim': torch.tensor(dst1_fim).float(),
            'tsf3_fim': torch.tensor(dst3_fim).float(),
            'fims1': torch.tensor(fims1_enc).float(),
            'fims3': torch.tensor(fims3_enc).float(),
            'T1': torch.tensor(T1).float(),
            'T3': torch.tensor(T3).float(),
            'j2d1': torch.tensor(pkl_data['j2d1']).float(),
            'j2d3': torch.tensor(pkl_data['j2d3']).float()
        }

        if 'warp1' in pkl_data:
            if len(pkl_data['warp1'].shape) == 4:
                sample['warp1'] = torch.tensor(pkl_data['warp1'][0], dtype=torch.float32)
                sample['warp3'] = torch.tensor(pkl_data['warp3'][0], dtype=torch.float32)
            else:
                sample['warp1'] = torch.tensor(pkl_data['warp1'], dtype=torch.float32)
                sample['warp3'] = torch.tensor(pkl_data['warp3'], dtype=torch.float32)
        elif 'warp_R' in pkl_data:
            sample['warp'] = torch.tensor(pkl_data['warp_R'][0], dtype=torch.float32)
        elif 'warp_T' in pkl_data:
            sample['warp'] = torch.tensor(pkl_data['warp_T'][0], dtype=torch.float32)

        if 'T1_cycle' in pkl_data:
            sample['T1_cycle'] = torch.tensor(pkl_data['T1_cycle']).float()
            sample['T3_cycle'] = torch.tensor(pkl_data['T3_cycle']).float()

        if 'T_cycle_vis' in pkl_data:
            sample['T_cycle_vis'] = torch.tensor(pkl_data['T_cycle_vis']).float()

        return sample

    def preprocess(self, sample):
        """
        Args:
           sample (dict): items contain
                --images (torch.FloatTensor): (2, 3, h, w)
                --fims (torch.FloatTensor): (2, 3, h, w)
                --T (torch.FloatTensor): (h, w, 2)
                --warp (torch.FloatTensor): (3, h, w)
                --head_bbox (torch.FloatTensor): (4), hear 4 = [lt_x, lt_y, rt_x, rt_y]

        Returns:
            new_sample (dict): items contain
                --src_inputs (torch.FloatTensor): (3+3, h, w)
                --tsf_inputs (torch.FloatTensor): (3+3, h, w)
                --T (torch.FloatTensor): (h, w, 2)
                --head_bbox (torch.FloatTensor): (4), hear 4 = [lt_x, lt_y, rt_x, rt_y]
                --images (torch.FloatTensor): (2, 3, h, w)
                --pseudo_masks (torch.FloatTensor) : (2, 1, h, w)
                --bg_inputs (torch.FloatTensor): (3+1, h, w) or (2, 3+1, h, w) if self.is_both is True
        """
        with torch.no_grad():
            images = sample['images']
            fims1 = sample['fims1']
            fims3 = sample['fims3']

            # 1. process the bg inputs
            src_fim = fims1[0]
            src_img = images[0]
            src_mask = src_fim[None, -1:, :, :]   # (1, h, w)
            src_bg_mask = morph(src_mask, ks=self.bg_ks, mode='erode', kernel=self.bg_kernel)[0]  # bg is 0, front is 1
            src_bg_inputs = torch.cat([src_img * src_bg_mask, src_bg_mask], dim=0)

            # 2. process the src inputs
            src_crop_mask = morph(src_mask, ks=self.ft_ks, mode='erode', kernel=self.ft_kernel)[0]
            src_inputs = torch.cat([src_img * (1 - src_crop_mask), src_fim])

            # 3. process the tsf inputs
            tsf1_fim = fims1[1]
            tsf1_mask = tsf1_fim[None, -1:, :, :]     # (1, h, w), bg is 0, front is 1
            tsf1_crop_mask = morph(tsf1_mask, ks=self.ft_ks, mode='erode', kernel=self.ft_kernel)[0]

            tsf3_fim = fims3[1]
            tsf3_mask = tsf3_fim[None, -1:, :, :]     # (1, h, w), bg is 0, front is 1
            tsf3_crop_mask = morph(tsf3_mask, ks=self.ft_ks, mode='erode', kernel=self.ft_kernel)[0]

            if 'warp1' not in sample or 'warp2' not in sample:
                warp1 = F.grid_sample(src_img[None], sample['T1'][None])[0]
                warp3 = F.grid_sample(src_img[None], sample['T3'][None])[0]
            else:
                warp1 = sample['warp1']
                warp3 = sample['warp3']
            tsf1_inputs = torch.cat([warp1, tsf1_fim], dim=0)
            tsf3_inputs = torch.cat([warp3, tsf3_fim], dim=0)

            if self.is_both:
                tsf1_img = images[1]
                tsf1_bg_mask = morph(tsf1_mask, ks=self.bg_ks, mode='dilate', kernel=self.bg_kernel)[0]  # bg is 0, front is 1
                tsf1_bg_inputs = torch.cat([tsf1_img * (1 - tsf1_bg_mask), tsf1_bg_mask], dim=0)

                tsf3_img = images[3]
                tsf3_bg_mask = morph(tsf3_mask, ks=self.bg_ks, mode='dilate', kernel=self.bg_kernel)[0]  # bg is 0, front is 1
                tsf3_bg_inputs = torch.cat([tsf3_img * (1 - tsf3_bg_mask), tsf3_bg_mask], dim=0)
                bg_inputs = torch.stack([src_bg_inputs, tsf1_bg_inputs, tsf3_bg_inputs], dim=0)
            else:
                bg_inputs = src_bg_inputs

            # 4. concat pseudo mask
            pseudo_masks = torch.stack([src_crop_mask, tsf1_crop_mask, tsf3_crop_mask], dim=0)

            new_sample = {
                'images': images,
                'pseudo_masks': pseudo_masks,
                'j2d1': sample['j2d1'],
                'j2d3': sample['j2d3'],
                'T1': sample['T1'],
                'T3': sample['T3'],
                'bg_inputs': bg_inputs,
                'src_inputs': src_inputs,
                'tsf1_inputs': tsf1_inputs,
                'tsf3_inputs': tsf3_inputs,
                'src_fim': sample['src_fim'],
                'tsf1_fim': sample['tsf1_fim'],
                'tsf3_fim': sample['tsf3_fim']
            }

            if 'T1_cycle' in sample:
                new_sample['T1_cycle'] = sample['T1_cycle']

            if 'T3_cycle' in sample:
                new_sample['T3_cycle'] = sample['T3_cycle']

            if 'T_cycle_vis' in sample:
                new_sample['T_cycle_vis'] = sample['T_cycle_vis']

            return new_sample





def make_dataset(opt):
    import platform

    class Config(object):
        pass

    config = Config()

    output_dir = opt.output_dir

    config.pair_ids_filepath = os.path.join(output_dir, 'pairs_meta.pkl')
    config.pkl_dir = os.path.join(output_dir, 'pairs')
    config.preds_img_folder = os.path.join(output_dir, 'imgs')
    config.image_size = opt.image_size
    config.map_name = opt.map_name
    config.uv_mapping = opt.uv_mapping
    config.is_both = False
    config.bg_ks = opt.bg_ks
    config.ft_ks = opt.ft_ks

    meta_cycle_ds = MetaCycleDataSet(opt=config)
    length = len(meta_cycle_ds)

    data_loader = torch.utils.data.DataLoader(
        meta_cycle_ds,
        batch_size=min(length, opt.batch_size),
        shuffle=False,
        num_workers=0 if platform.system() == 'Windows' else 4,
        drop_last=True)

    return data_loader


def adaptive_personalize(opt, imitator, visualizer):
    output_dir = opt.output_dir
    mkdirs([os.path.join(output_dir, 'imgs'), os.path.join(output_dir, 'pairs')])

    # TODO check if it has been computed.
    print('\n\t\t\tPersonalization: meta imitation...')
    imitator.personalize(opt.src_path, visualizer=None)
    meta_imitate(opt, imitator, prior_tgt_path=opt.pri_path, visualizer=None, save_imgs=True)

    # post tune
    print('\n\t\t\tPersonalization: meta cycle finetune...')
    loader = make_dataset(opt)
    imitator.post_personalize(opt.output_dir, loader, visualizer=None, verbose=False)


if __name__ == "__main__":
    # meta imitator
    test_opt = TestOptions().parse()

    if test_opt.ip:
        visualizer = VisdomVisualizer(env=test_opt.name, ip=test_opt.ip, port=test_opt.port)
    else:
        visualizer = None

    # set imitator
    imitator = Imitator(test_opt)

    if test_opt.post_tune:
        adaptive_personalize(test_opt, imitator, visualizer)

    imitator.personalize(test_opt.src_path, visualizer=visualizer)
    print('\n\t\t\tPersonalization: completed...')

    if test_opt.save_res:
        pred_output_dir = mkdir(os.path.join(test_opt.output_dir, 'imitators'))
        pred_output_dir = clear_dir(pred_output_dir)
    else:
        pred_output_dir = None

    print('\n\t\t\tImitating `{}`'.format(test_opt.tgt_path))
    tgt_paths1 = scan_tgt_paths(test_opt.tgt_path, itv=1, start=0)
    tgt_paths2 = scan_tgt_paths(test_opt.tgt_path, itv=1, start=1)
    tgt_paths3 = scan_tgt_paths(test_opt.tgt_path, itv=1, start=2)
    tgt_paths = [(tgt_paths1[i], tgt_paths2[i], tgt_paths3[i]) for i in range(len(tgt_paths1))]
    imitator.inference(tgt_paths, tgt_smpls=None, cam_strategy='smooth',
                       output_dir=pred_output_dir, visualizer=visualizer, verbose=True)



