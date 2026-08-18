import random
import torch
import torch.utils.data
from lib.utils import TensorDict
import numpy as np


def no_processing(data):
    return data


class TrackingSamplerThreeMDOT(torch.utils.data.Dataset):
    """ Class responsible for sampling frames from training sequences to form batches.

    The sampling is done in the following ways. First a dataset is selected at random. Next, a sequence is selected
    from that dataset. A base frame is then sampled randomly from the sequence. Next, a set of 'train frames' and
    'test frames' are sampled from the sequence from the range [base_frame_id - max_gap, base_frame_id]  and
    (base_frame_id, base_frame_id + max_gap] respectively. Only the frames in which the target is visible are sampled.
    If enough visible frames are not found, the 'max_gap' is increased gradually till enough frames are found.

    The sampled frames are then passed through the input 'processing' function for the necessary processing-
    """

    def __init__(self, datasets, p_datasets, samples_per_epoch, max_gap,
                 num_search_frames, num_template_frames=1, processing=no_processing, frame_sample_mode='causal',
                 train_cls=False, pos_prob=0.5, require_all_views_visible=False,
                 canonical_view_order=False, max_retry=500, debug_exceptions=False):
        """
        args:
            datasets - List of datasets to be used for training
            p_datasets - List containing the probabilities by which each dataset will be sampled
            samples_per_epoch - Number of training samples per epoch
            max_gap - Maximum gap, in frame numbers, between the train frames and the test frames.
            num_search_frames - Number of search frames to sample.
            num_template_frames - Number of template frames to sample.
            processing - An instance of Processing class which performs the necessary processing of the data.
            frame_sample_mode - Either 'causal' or 'interval'. If 'causal', then the test frames are sampled in a causally,
                                otherwise randomly within the interval.
        """
        self.datasets = datasets
        self.train_cls = train_cls  # whether we are training classification
        self.pos_prob = pos_prob  # probability of sampling positive class when making classification

        # If p not provided, sample uniformly from all videos
        if p_datasets is None:
            p_datasets = [len(d) for d in self.datasets]

        # Normalize
        p_total = sum(p_datasets)
        self.p_datasets = [x / p_total for x in p_datasets]

        self.samples_per_epoch = samples_per_epoch
        self.max_gap = max_gap
        self.num_search_frames = num_search_frames
        self.num_template_frames = num_template_frames
        self.processing = processing
        self.frame_sample_mode = frame_sample_mode
        self.require_all_views_visible = bool(require_all_views_visible)
        self.canonical_view_order = bool(canonical_view_order)
        self.max_retry = int(max_retry)
        self.debug_exceptions = bool(debug_exceptions)

    def __len__(self):
        return self.samples_per_epoch

    def _sample_visible_ids(self, visible, num_ids=1, min_id=None, max_id=None,
                            allow_invisible=False, force_invisible=False):
        """ Samples num_ids frames between min_id and max_id for which target is visible

        args:
            visible - 1d Tensor indicating whether target is visible for each frame
            num_ids - number of frames to be samples
            min_id - Minimum allowed frame number
            max_id - Maximum allowed frame number

        returns:
            list - List of sampled frame numbers. None if not sufficient visible frames could be found.
        """
        if num_ids == 0:
            return []
        if min_id is None or min_id < 0:
            min_id = 0
        if max_id is None or max_id > len(visible):
            max_id = len(visible)
        # get valid ids
        if force_invisible:
            valid_ids = [i for i in range(min_id, max_id) if not visible[i]]
        else:
            if allow_invisible:
                valid_ids = [i for i in range(min_id, max_id)]
            else:
                valid_ids = [i for i in range(min_id, max_id) if visible[i]]

        # No visible ids
        if len(valid_ids) == 0:
            return None

        return random.choices(valid_ids, k=num_ids)

    def _ids_visible(self, visible, ids):
        if ids is None:
            return False
        if isinstance(ids, int):
            ids = [ids]
        for frame_id in ids:
            if frame_id is None or frame_id < 0 or frame_id >= len(visible):
                return False
            if not bool(visible[frame_id]):
                return False
        return True

    def _common_visible(self, visible_list):
        if len(visible_list) == 0:
            return None
        min_len = min(len(v) for v in visible_list)
        if min_len <= 0:
            return None

        common = torch.as_tensor(visible_list[0][:min_len]).bool().clone()
        for visible in visible_list[1:]:
            common = common & torch.as_tensor(visible[:min_len]).bool()
        return common

    def _view_visible_flags(self, visible_list, frame_ids):
        flags = []
        for visible in visible_list:
            flags.append(self._ids_visible(visible, frame_ids))
        return torch.tensor(flags, dtype=torch.bool)

    def _print_exception(self, msg=""):
        if not self.debug_exceptions:
            return
        import traceback
        print("\n" + "=" * 80)
        print("[TrackingSamplerThreeMDOT] Sampling failed:", msg)
        traceback.print_exc()
        print("=" * 80 + "\n")

    def _view_order_key(self, dataset, seq_id):
        name = dataset.sequence_list[seq_id]
        try:
            return int(name.rsplit("-", 1)[-1])
        except Exception:
            return name

    @staticmethod
    def _target_and_view(sequence_name):
        target_id, raw_view = sequence_name.rsplit("-", 1)
        view_map = {"1": "A", "2": "B", "3": "C"}
        return target_id, view_map.get(raw_view, raw_view)

    def _canonicalize_views(self, dataset, seq_id, seq_info_dict, visible,
                            rest_seq_id_list, rest_info_list, rest_visible_list):
        if not self.canonical_view_order:
            return seq_id, seq_info_dict, visible, rest_seq_id_list, rest_info_list, rest_visible_list

        view_records = [(seq_id, seq_info_dict, visible)]
        view_records += list(zip(rest_seq_id_list, rest_info_list, rest_visible_list))
        view_records = sorted(view_records, key=lambda item: self._view_order_key(dataset, item[0]))

        seq_id, seq_info_dict, visible = view_records[0]
        rest_records = view_records[1:]
        rest_seq_id_list = [item[0] for item in rest_records]
        rest_info_list = [item[1] for item in rest_records]
        rest_visible_list = [item[2] for item in rest_records]

        return seq_id, seq_info_dict, visible, rest_seq_id_list, rest_info_list, rest_visible_list

    def __getitem__(self, index):
        if self.train_cls:
            return self.getitem_cls()
        else:
            return self.getitem()

    def getitem(self):
        """
        returns:
            TensorDict - dict containing all the data blocks
        """
        valid = False
        retry_count = 0
        while not valid:
            retry_count += 1
            if retry_count > self.max_retry:
                raise RuntimeError(
                    "TrackingSamplerThreeMDOT failed after %d retries. "
                    "Check ThreeMDOT visibility labels or disable "
                    "TRAIN.PCUM.REQUIRE_ALL_VIEWS_VISIBLE." % self.max_retry
                )

            # Select a dataset
            dataset = random.choices(self.datasets, self.p_datasets)[0]

            is_video_dataset = dataset.is_video_sequence()

            # sample a sequence from the given dataset
            seq_id, visible, seq_info_dict = self.sample_seq_from_dataset(dataset, is_video_dataset)     # 从数据集中随机挑选了一个sequence

            rest_seq_id_list, rest_visible_list, rest_seq_info_dict_list = self.sample_seq_from_dataset_threemdot(dataset, is_video_dataset, seq_id)     # 选取对应的sequence

            seq_id, seq_info_dict, visible, rest_seq_id_list, rest_seq_info_dict_list, rest_visible_list = \
                self._canonicalize_views(
                    dataset,
                    seq_id,
                    seq_info_dict,
                    visible,
                    rest_seq_id_list,
                    rest_seq_info_dict_list,
                    rest_visible_list,
                )

            if self.require_all_views_visible and len(rest_visible_list) < 2:
                continue

            visible_for_sampling = visible
            valid_for_sampling = seq_info_dict.get("valid", visible)
            if self.require_all_views_visible and is_video_dataset:
                common_visible = self._common_visible([visible] + rest_visible_list)
                if common_visible is None:
                    continue
                visible_for_sampling = common_visible
                valid_for_sampling = common_visible

            if is_video_dataset:
                template_frame_ids = None
                search_frame_ids = None
                gap_increase = 0

                if self.frame_sample_mode == 'causal':
                    # Sample test and train frames in a causal manner, i.e. search_frame_ids > template_frame_ids
                    while search_frame_ids is None:
                        base_frame_id = self._sample_visible_ids(visible_for_sampling, num_ids=1, min_id=self.num_template_frames - 1,
                                                                 max_id=len(visible_for_sampling) - self.num_search_frames)
                        if base_frame_id is None:
                            break
                        prev_frame_ids = self._sample_visible_ids(visible_for_sampling, num_ids=self.num_template_frames - 1,
                                                                  min_id=base_frame_id[0] - self.max_gap - gap_increase,
                                                                  max_id=base_frame_id[0])
                        if prev_frame_ids is None:
                            gap_increase += 5
                            continue
                        template_frame_ids = base_frame_id + prev_frame_ids
                        search_frame_ids = self._sample_visible_ids(visible_for_sampling, min_id=template_frame_ids[0] + 1,
                                                                  max_id=template_frame_ids[0] + self.max_gap + gap_increase,
                                                                  num_ids=self.num_search_frames)
                        # Increase gap until a frame is found
                        gap_increase += 5

                elif self.frame_sample_mode == "trident" or self.frame_sample_mode == "trident_pro":
                    template_frame_ids, search_frame_ids = self.get_frame_ids_trident(visible_for_sampling)
                elif self.frame_sample_mode == "stark":
                    template_frame_ids, search_frame_ids = self.get_frame_ids_stark(visible_for_sampling, valid_for_sampling)
                else:
                    raise ValueError("Illegal frame sample mode")

                if template_frame_ids is None or search_frame_ids is None:
                    continue
            else:
                # In case of image dataset, just repeat the image to generate synthetic video
                template_frame_ids = [1] * self.num_template_frames
                search_frame_ids = [1] * self.num_search_frames

            template_view_valid = self._view_visible_flags([visible] + rest_visible_list, template_frame_ids)
            search_view_valid = self._view_visible_flags([visible] + rest_visible_list, search_frame_ids)
            if self.require_all_views_visible:
                if not bool(template_view_valid.all()) or not bool(search_view_valid.all()):
                    continue

            try:
                sequence_ids = [seq_id] + list(rest_seq_id_list)
                target_view_pairs = [
                    self._target_and_view(dataset.sequence_list[item])
                    for item in sequence_ids
                ]
                target_ids = {item[0] for item in target_view_pairs}
                if len(target_ids) != 1:
                    raise RuntimeError(
                        "ThreeMDOT group mixes target ids: %s" %
                        sorted(target_ids))
                template_frames, template_anno, meta_obj_train = dataset.get_frames(seq_id, template_frame_ids, seq_info_dict)
                search_frames, search_anno, meta_obj_test = dataset.get_frames(seq_id, search_frame_ids, seq_info_dict)

                for i in range(0, len(rest_seq_id_list)):    # 三个模板加到一起
                    tmp_template_frames, tmp_template_anno, tmp_meta_obj_train = dataset.get_frames(rest_seq_id_list[i], template_frame_ids, rest_seq_info_dict_list[i])
                    template_frames = template_frames + tmp_template_frames
                    template_anno['bbox'] = template_anno['bbox'] + tmp_template_anno['bbox']
                    tmp_search_frames, tmp_search_anno, _ = dataset.get_frames(
                        rest_seq_id_list[i], search_frame_ids, rest_seq_info_dict_list[i])
                    search_frames = search_frames + tmp_search_frames
                    search_anno['bbox'] = search_anno['bbox'] + tmp_search_anno['bbox']


                H, W, _ = template_frames[0].shape
                template_masks = template_anno['mask'] if 'mask' in template_anno else [torch.zeros((H, W))] * self.num_template_frames*3      # 双机翻两倍
                search_masks = search_anno['mask'] if 'mask' in search_anno else [torch.zeros((H, W))] * self.num_search_frames*3

                data = TensorDict({'template_images': template_frames,
                                   'template_anno': template_anno['bbox'],
                                   'template_masks': template_masks,
                                   'search_images': search_frames,
                                   'search_anno': search_anno['bbox'],
                                   'search_masks': search_masks,
                                   'template_view_valid': template_view_valid,
                                   'search_view_valid': search_view_valid,
                                   'target_id': target_view_pairs[0][0],
                                   'view_ids': [item[1] for item in target_view_pairs],
                                   'template_frame_ids': list(template_frame_ids),
                                   'search_frame_ids': list(search_frame_ids),
                                   'dataset': dataset.get_name(),
                                   'test_class': meta_obj_test.get('object_class_name')})
                # make data augmentation
                data = self.processing(data)

                # check whether data is valid
                valid = data['valid']
            except:
                self._print_exception("getitem try block failed")
                valid = False
        return data

    def getitem_cls(self):
        # get data for classification
        """
        args:
            index (int): Index (Ignored since we sample randomly)
            aux (bool): whether the current data is for auxiliary use (e.g. copy-and-paste)

        returns:
            TensorDict - dict containing all the data blocks
        """
        valid = False
        label = None
        while not valid:
            # Select a dataset
            dataset = random.choices(self.datasets, self.p_datasets)[0]

            is_video_dataset = dataset.is_video_sequence()

            # sample a sequence from the given dataset
            seq_id, visible, seq_info_dict = self.sample_seq_from_dataset(dataset, is_video_dataset)
            # sample template and search frame ids
            if is_video_dataset:
                if self.frame_sample_mode in ["trident", "trident_pro"]:
                    template_frame_ids, search_frame_ids = self.get_frame_ids_trident(visible)
                elif self.frame_sample_mode == "stark":
                    template_frame_ids, search_frame_ids = self.get_frame_ids_stark(visible, seq_info_dict["valid"])
                else:
                    raise ValueError("illegal frame sample mode")
            else:
                # In case of image dataset, just repeat the image to generate synthetic video
                template_frame_ids = [1] * self.num_template_frames
                search_frame_ids = [1] * self.num_search_frames
            try:
                # "try" is used to handle trackingnet data failure
                # get images and bounding boxes (for templates)
                template_frames, template_anno, meta_obj_train = dataset.get_frames(seq_id, template_frame_ids,
                                                                                    seq_info_dict)
                H, W, _ = template_frames[0].shape
                template_masks = template_anno['mask'] if 'mask' in template_anno else [torch.zeros(
                    (H, W))] * self.num_template_frames
                # get images and bounding boxes (for searches)
                # positive samples
                if random.random() < self.pos_prob:
                    label = torch.ones(1,)
                    search_frames, search_anno, meta_obj_test = dataset.get_frames(seq_id, search_frame_ids, seq_info_dict)
                    search_masks = search_anno['mask'] if 'mask' in search_anno else [torch.zeros(
                        (H, W))] * self.num_search_frames
                # negative samples
                else:
                    label = torch.zeros(1,)
                    if is_video_dataset:
                        search_frame_ids = self._sample_visible_ids(visible, num_ids=1, force_invisible=True)
                        if search_frame_ids is None:
                            search_frames, search_anno, meta_obj_test = self.get_one_search()
                        else:
                            search_frames, search_anno, meta_obj_test = dataset.get_frames(seq_id, search_frame_ids,
                                                                                           seq_info_dict)
                            search_anno["bbox"] = [self.get_center_box(H, W)]
                    else:
                        search_frames, search_anno, meta_obj_test = self.get_one_search()
                    H, W, _ = search_frames[0].shape
                    search_masks = search_anno['mask'] if 'mask' in search_anno else [torch.zeros(
                        (H, W))] * self.num_search_frames

                data = TensorDict({'template_images': template_frames,
                                   'template_anno': template_anno['bbox'],
                                   'template_masks': template_masks,
                                   'search_images': search_frames,
                                   'search_anno': search_anno['bbox'],
                                   'search_masks': search_masks,
                                   'dataset': dataset.get_name(),
                                   'test_class': meta_obj_test.get('object_class_name')})

                # make data augmentation
                data = self.processing(data)
                # add classification label
                data["label"] = label
                # check whether data is valid
                valid = data['valid']
            except:
                valid = False

        return data

    def get_center_box(self, H, W, ratio=1/8):
        cx, cy, w, h = W/2, H/2, W * ratio, H * ratio
        return torch.tensor([int(cx-w/2), int(cy-h/2), int(w), int(h)])


    # 把sample足够visible的去掉了，因为双机本来就要处理遮挡
    def sample_seq_from_dataset(self, dataset, is_video_dataset):

        # Sample a sequence with enough visible frames
        # enough_visible_frames = False
        # while not enough_visible_frames:
            # Sample a sequence
        seq_id = random.randint(0, dataset.get_num_sequences() - 1)

        # Sample frames
        seq_info_dict = dataset.get_sequence_info(seq_id)
        visible = seq_info_dict['visible']

            # enough_visible_frames = visible.type(torch.int64).sum().item() > 2 * (
            #         self.num_search_frames + self.num_template_frames) and len(visible) >= 20

            # enough_visible_frames = enough_visible_frames or not is_video_dataset
        return seq_id, visible, seq_info_dict


    # 得到另一机
    def sample_seq_from_dataset_threemdot(self, dataset, is_video_dataset, seq_id):

        seq_cls = dataset.sequence_list[seq_id][:-2]       # 'md20xx'

        seq_cls_list = dataset.seq_per_class[seq_cls]

        seq_cls_list_rest = seq_cls_list.copy()

        seq_cls_list_rest.remove(seq_id)

        rest_seq_id_list = []
        rest_visible_list = []
        rest_seq_info_dict_list = []

        for i, id in enumerate(seq_cls_list_rest):
            rest_seq_id_list.append(id)
            tmp_info = dataset.get_sequence_info(id)
            rest_seq_info_dict_list.append(tmp_info)
            rest_visible_list.append(tmp_info['visible'])



        # Sample a sequence with enough visible frames

        # Sample frames




        return rest_seq_id_list, rest_visible_list, rest_seq_info_dict_list


    def get_one_search(self):
        # Select a dataset
        dataset = random.choices(self.datasets, self.p_datasets)[0]

        is_video_dataset = dataset.is_video_sequence()
        # sample a sequence
        seq_id, visible, seq_info_dict = self.sample_seq_from_dataset(dataset, is_video_dataset)
        # sample a frame
        if is_video_dataset:
            if self.frame_sample_mode == "stark":
                search_frame_ids = self._sample_visible_ids(seq_info_dict["valid"], num_ids=1)
            else:
                search_frame_ids = self._sample_visible_ids(visible, num_ids=1, allow_invisible=True)
        else:
            search_frame_ids = [1]
        # get the image, bounding box and other info
        search_frames, search_anno, meta_obj_test = dataset.get_frames(seq_id, search_frame_ids, seq_info_dict)

        return search_frames, search_anno, meta_obj_test

    def get_frame_ids_trident(self, visible):
        # get template and search ids in a 'trident' manner
        template_frame_ids_extra = []
        while None in template_frame_ids_extra or len(template_frame_ids_extra) == 0:
            template_frame_ids_extra = []
            # first randomly sample two frames from a video
            template_frame_id1 = self._sample_visible_ids(visible, num_ids=1)  # the initial template id
            search_frame_ids = self._sample_visible_ids(visible, num_ids=1)  # the search region id
            # get the dynamic template id
            for max_gap in self.max_gap:
                if template_frame_id1[0] >= search_frame_ids[0]:
                    min_id, max_id = search_frame_ids[0], search_frame_ids[0] + max_gap
                else:
                    min_id, max_id = search_frame_ids[0] - max_gap, search_frame_ids[0]
                if self.frame_sample_mode == "trident_pro":
                    f_id = self._sample_visible_ids(visible, num_ids=1, min_id=min_id, max_id=max_id,
                                                    allow_invisible=True)
                else:
                    f_id = self._sample_visible_ids(visible, num_ids=1, min_id=min_id, max_id=max_id)
                if f_id is None:
                    template_frame_ids_extra += [None]
                else:
                    template_frame_ids_extra += f_id

        template_frame_ids = template_frame_id1 + template_frame_ids_extra
        return template_frame_ids, search_frame_ids

    def get_frame_ids_stark(self, visible, valid):
        # get template and search ids in a 'stark' manner
        template_frame_ids_extra = []
        while None in template_frame_ids_extra or len(template_frame_ids_extra) == 0:
            template_frame_ids_extra = []
            # first randomly sample two frames from a video
            template_frame_id1 = self._sample_visible_ids(visible, num_ids=1)  # the initial template id
            search_frame_ids = self._sample_visible_ids(visible, num_ids=1)  # the search region id
            # get the dynamic template id
            for max_gap in self.max_gap:
                if template_frame_id1[0] >= search_frame_ids[0]:
                    min_id, max_id = search_frame_ids[0], search_frame_ids[0] + max_gap
                else:
                    min_id, max_id = search_frame_ids[0] - max_gap, search_frame_ids[0]
                """we require the frame to be valid but not necessary visible"""
                f_id = self._sample_visible_ids(valid, num_ids=1, min_id=min_id, max_id=max_id)
                if f_id is None:
                    template_frame_ids_extra += [None]
                else:
                    template_frame_ids_extra += f_id

        template_frame_ids = template_frame_id1 + template_frame_ids_extra
        return template_frame_ids, search_frame_ids
