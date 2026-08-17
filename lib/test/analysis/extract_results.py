import os
import sys
import numpy as np
from lib.test.utils.load_text import load_text
import torch
import pickle
from tqdm import tqdm
try:
    import tikzplotlib
except ModuleNotFoundError:
    class _TikzPlotlibFallback:
        @staticmethod
        def save(*args, **kwargs):
            return None

    tikzplotlib = _TikzPlotlibFallback()
import matplotlib
import matplotlib.pyplot as plt
import json

env_path = os.path.join(os.path.dirname(__file__), '../../..')
if env_path not in sys.path:
    sys.path.append(env_path)

from lib.test.evaluation.environment import env_settings
from lib.test.evaluation.run_id import format_run_id


def calc_err_center(pred_bb, anno_bb, normalized=False):
    pred_center = pred_bb[:, :2] + 0.5 * (pred_bb[:, 2:] - 1.0)
    anno_center = anno_bb[:, :2] + 0.5 * (anno_bb[:, 2:] - 1.0)

    if normalized:
        pred_center = pred_center / anno_bb[:, 2:]
        anno_center = anno_center / anno_bb[:, 2:]

    err_center = ((pred_center - anno_center)**2).sum(1).sqrt()
    return err_center


def calc_iou_overlap(pred_bb, anno_bb):
    tl = torch.max(pred_bb[:, :2], anno_bb[:, :2])
    br = torch.min(pred_bb[:, :2] + pred_bb[:, 2:] - 1.0, anno_bb[:, :2] + anno_bb[:, 2:] - 1.0)
    sz = (br - tl + 1.0).clamp(0)

    # Area
    intersection = sz.prod(dim=1)
    union = pred_bb[:, 2:].prod(dim=1) + anno_bb[:, 2:].prod(dim=1) - intersection

    return intersection / union


def calc_seq_err_robust(pred_bb, anno_bb, dataset, target_visible=None):
    pred_bb = pred_bb.clone()

    # Check if invalid values are present
    if torch.isnan(pred_bb).any() or (pred_bb[:, 2:] < 0.0).any():
        raise Exception('Error: Invalid results')

    if torch.isnan(anno_bb).any():
        if dataset == 'uav':
            pass
        else:
            raise Exception('Warning: NaNs in annotation')

    if (pred_bb[:, 2:] == 0.0).any():
        for i in range(1, pred_bb.shape[0]):
            if (pred_bb[i, 2:] == 0.0).any() and not torch.isnan(anno_bb[i, :]).any():
                pred_bb[i, :] = pred_bb[i-1, :]

    if pred_bb.shape[0] != anno_bb.shape[0]:
        if dataset == 'lasot':
            if pred_bb.shape[0] > anno_bb.shape[0]:
                # For monkey-17, there is a mismatch for some trackers.
                pred_bb = pred_bb[:anno_bb.shape[0], :]
            else:
                raise Exception('Mis-match in tracker prediction and GT lengths')
        else:
            if pred_bb.shape[0] > anno_bb.shape[0]:
                pred_bb = pred_bb[:anno_bb.shape[0], :]
            else:
                pad = torch.zeros((anno_bb.shape[0] - pred_bb.shape[0], 4)).type_as(pred_bb)
                pred_bb = torch.cat((pred_bb, pad), dim=0)

    pred_bb[0, :] = anno_bb[0, :]

    if target_visible is not None:
        target_visible = target_visible.bool()
        valid = ((anno_bb[:, 2:] > 0.0).sum(1) == 2) & target_visible
    else:
        valid = ((anno_bb[:, 2:] > 0.0).sum(1) == 2)

    err_center = calc_err_center(pred_bb, anno_bb)
    err_center_normalized = calc_err_center(pred_bb, anno_bb, normalized=True)
    err_overlap = calc_iou_overlap(pred_bb, anno_bb)

    # handle invalid anno cases
    if dataset in ['uav']:
        err_center[~valid] = -1.0
    else:
        err_center[~valid] = float("Inf")
    err_center_normalized[~valid] = -1.0
    err_overlap[~valid] = -1.0

    if dataset == 'lasot' and target_visible is not None:
        err_center_normalized[~target_visible] = float("Inf")
        err_center[~target_visible] = float("Inf")

    if torch.isnan(err_overlap).any():
        raise Exception('Nans in calculated overlap')
    return err_overlap, err_center, err_center_normalized, valid


def extract_results(trackers, dataset, report_name, skip_missing_seq=False, plot_bin_gap=0.05, exclude_invalid_frames=False):
    settings = env_settings()
    result_plot_path = os.path.join(settings.result_plot_path, report_name)

    if not os.path.exists(result_plot_path):
        os.makedirs(result_plot_path)

    threshold_set_overlap = torch.arange(0.0, 1.0 + plot_bin_gap, plot_bin_gap, dtype=torch.float64)
    threshold_set_center = torch.arange(0, 51, dtype=torch.float64)
    threshold_set_center_norm = torch.arange(0, 51, dtype=torch.float64) / 100.0

    avg_overlap_all = torch.zeros((len(dataset), len(trackers)), dtype=torch.float64)
    ave_success_rate_plot_overlap = torch.zeros((len(dataset), len(trackers), threshold_set_overlap.numel()), dtype=torch.float32)
    ave_success_rate_plot_center = torch.zeros((len(dataset), len(trackers), threshold_set_center.numel()), dtype=torch.float32)
    ave_success_rate_plot_center_norm = torch.zeros((len(dataset), len(trackers), threshold_set_center.numel()), dtype=torch.float32)

    valid_sequence = torch.ones(len(dataset), dtype=torch.uint8)

    for seq_id, seq in enumerate(tqdm(dataset)):
        anno_bb = torch.tensor(seq.ground_truth_rect)
        target_visible = torch.tensor(seq.target_visible, dtype=torch.uint8) if getattr(seq, 'target_visible', None) is not None else None
        
        for trk_id, trk in enumerate(trackers):
            base_results_path = '{}/{}'.format(trk.results_dir, seq.name)
            results_path = '{}.txt'.format(base_results_path)

            if os.path.isfile(results_path):
                pred_bb = torch.tensor(load_text(str(results_path), delimiter=('\t', ','), dtype=np.float64))
            else:
                if skip_missing_seq:
                    valid_sequence[seq_id] = 0
                    break
                else:
                    raise Exception('Result not found. {}'.format(results_path))

            err_overlap, err_center, err_center_normalized, valid_frame = calc_seq_err_robust(
                pred_bb, anno_bb, seq.dataset, target_visible)

            avg_overlap_all[seq_id, trk_id] = err_overlap[valid_frame].mean()

            if exclude_invalid_frames:
                seq_length = valid_frame.long().sum()
            else:
                seq_length = anno_bb.shape[0]

            if seq_length <= 0:
                raise Exception('Seq length zero')

            ave_success_rate_plot_overlap[seq_id, trk_id, :] = (err_overlap.view(-1, 1) > threshold_set_overlap.view(1, -1)).sum(0).float() / seq_length
            ave_success_rate_plot_center[seq_id, trk_id, :] = (err_center.view(-1, 1) <= threshold_set_center.view(1, -1)).sum(0).float() / seq_length
            ave_success_rate_plot_center_norm[seq_id, trk_id, :] = (err_center_normalized.view(-1, 1) <= threshold_set_center_norm.view(1, -1)).sum(0).float() / seq_length

    print('\n\nComputed results over {} / {} sequences'.format(valid_sequence.long().sum().item(), valid_sequence.shape[0]))

    seq_names = [s.name for s in dataset]
    tracker_names = [{'name': t.name, 'param': t.parameter_name, 'run_id': t.run_id, 'disp_name': t.display_name} for t in trackers]

    eval_data = {'sequences': seq_names, 'trackers': tracker_names,
                 'valid_sequence': valid_sequence.tolist(),
                 'ave_success_rate_plot_overlap': ave_success_rate_plot_overlap.tolist(),
                 'ave_success_rate_plot_center': ave_success_rate_plot_center.tolist(),
                 'ave_success_rate_plot_center_norm': ave_success_rate_plot_center_norm.tolist(),
                 'avg_overlap_all': avg_overlap_all.tolist(),
                 'threshold_set_overlap': threshold_set_overlap.tolist(),
                 'threshold_set_center': threshold_set_center.tolist(),
                 'threshold_set_center_norm': threshold_set_center_norm.tolist()}

    with open(result_plot_path + '/eval_data.pkl', 'wb') as fh:
        pickle.dump(eval_data, fh)

    return eval_data


def get_plot_draw_styles():
    plot_draw_style = [{'color': (1.0, 0.0, 0.0), 'line_style': '-'},
                       {'color': (0.0, 1.0, 0.0), 'line_style': '-'},
                       {'color': (0.0, 0.0, 1.0), 'line_style': '-'},
                       {'color': (0.0, 0.0, 0.0), 'line_style': '-'},
                       {'color': (1.0, 0.0, 1.0), 'line_style': '-'},
                       {'color': (0.0, 1.0, 1.0), 'line_style': '-'},
                       {'color': (0.5, 0.5, 0.5), 'line_style': '-'},
                       {'color': (136.0 / 255.0, 0.0, 21.0 / 255.0), 'line_style': '-'},
                       {'color': (1.0, 127.0 / 255.0, 39.0 / 255.0), 'line_style': '-'},
                       {'color': (0.0, 162.0 / 255.0, 232.0 / 255.0), 'line_style': '-'},
                       {'color': (0.0, 0.5, 0.0), 'line_style': '-'},
                       {'color': (1.0, 0.5, 0.2), 'line_style': '-'},
                       {'color': (0.1, 0.4, 0.0), 'line_style': '-'},
                       {'color': (0.6, 0.3, 0.9), 'line_style': '-'},
                       {'color': (0.4, 0.7, 0.1), 'line_style': '-'},
                       {'color': (0.2, 0.1, 0.7), 'line_style': '-'},
                       {'color': (0.7, 0.6, 0.2), 'line_style': '-'}]

    return plot_draw_style


def check_eval_data_is_valid(eval_data, trackers, dataset):
    """ Checks if the pre-computed results are valid"""
    seq_names = [s.name for s in dataset]
    seq_names_saved = eval_data['sequences']

    tracker_names_f = [(t.name, t.parameter_name, t.run_id) for t in trackers]
    tracker_names_f_saved = [(t['name'], t['param'], t['run_id']) for t in eval_data['trackers']]

    return seq_names == seq_names_saved and tracker_names_f == tracker_names_f_saved


def merge_multiple_runs(eval_data):
    new_tracker_names = []
    ave_success_rate_plot_overlap_merged = []
    ave_success_rate_plot_center_merged = []
    ave_success_rate_plot_center_norm_merged = []
    avg_overlap_all_merged = []

    ave_success_rate_plot_overlap = torch.tensor(eval_data['ave_success_rate_plot_overlap'])
    ave_success_rate_plot_center = torch.tensor(eval_data['ave_success_rate_plot_center'])
    ave_success_rate_plot_center_norm = torch.tensor(eval_data['ave_success_rate_plot_center_norm'])
    avg_overlap_all = torch.tensor(eval_data['avg_overlap_all'])

    trackers = eval_data['trackers']
    merged = torch.zeros(len(trackers), dtype=torch.uint8)
    for i in range(len(trackers)):
        if merged[i]:
            continue
        base_tracker = trackers[i]
        new_tracker_names.append(base_tracker)

        # 💡【核心修复】：通过 name, param 和 disp_name 精确匹配，防止把不同视角的曲线强行平均成一条
        match = [t['name'] == base_tracker['name'] and 
                 t['param'] == base_tracker['param'] and 
                 t.get('disp_name') == base_tracker.get('disp_name') for t in trackers]
        match = torch.tensor(match)

        ave_success_rate_plot_overlap_merged.append(ave_success_rate_plot_overlap[:, match, :].mean(1))
        ave_success_rate_plot_center_merged.append(ave_success_rate_plot_center[:, match, :].mean(1))
        ave_success_rate_plot_center_norm_merged.append(ave_success_rate_plot_center_norm[:, match, :].mean(1))
        avg_overlap_all_merged.append(avg_overlap_all[:, match].mean(1))

        merged[match] = 1

    ave_success_rate_plot_overlap_merged = torch.stack(ave_success_rate_plot_overlap_merged, dim=1)
    ave_success_rate_plot_center_merged = torch.stack(ave_success_rate_plot_center_merged, dim=1)
    ave_success_rate_plot_center_norm_merged = torch.stack(ave_success_rate_plot_center_norm_merged, dim=1)
    avg_overlap_all_merged = torch.stack(avg_overlap_all_merged, dim=1)

    eval_data['trackers'] = new_tracker_names
    eval_data['ave_success_rate_plot_overlap'] = ave_success_rate_plot_overlap_merged.tolist()
    eval_data['ave_success_rate_plot_center'] = ave_success_rate_plot_center_merged.tolist()
    eval_data['ave_success_rate_plot_center_norm'] = ave_success_rate_plot_center_norm_merged.tolist()
    eval_data['avg_overlap_all'] = avg_overlap_all_merged.tolist()

    return eval_data


def get_tracker_display_name(tracker):
    if tracker.get('disp_name') is None:
        if tracker.get('run_id') is None:
            disp_name = '{}_{}'.format(tracker['name'], tracker['param'])
        else:
            disp_name = '{}_{}_{}'.format(
                tracker['name'], tracker['param'],
                format_run_id(tracker['run_id']))
    else:
        disp_name = tracker['disp_name']
    return disp_name


def plot_draw_save(y, x, scores, trackers, plot_draw_styles, result_plot_path, plot_opts):
    plt.rcParams['text.usetex']=False  # 如果你没有装 LaTeX，设为 False 防止报错
    plt.rcParams["font.family"] = "Times New Roman"
    
    font_size = plot_opts.get('font_size', 20)
    font_size_axis = plot_opts.get('font_size_axis', 20)
    line_width = plot_opts.get('line_width', 2)
    font_size_legend = plot_opts.get('font_size_legend', 20)

    plot_type = plot_opts['plot_type']
    legend_loc = plot_opts['legend_loc']

    xlabel = plot_opts['xlabel']
    ylabel = plot_opts['ylabel']
    xlim = plot_opts['xlim']
    ylim = plot_opts['ylim']
    title = plot_opts['title']

    matplotlib.rcParams.update({'font.size': font_size})
    matplotlib.rcParams.update({'axes.titlesize': font_size_axis})
    matplotlib.rcParams.update({'axes.titleweight': 'black'})
    matplotlib.rcParams.update({'axes.labelsize': font_size_axis})

    fig, ax = plt.subplots()
    index_sort = scores.argsort(descending=False)

    plotted_lines = []
    legend_text = []

    for id, id_sort in enumerate(index_sort):
        line = ax.plot(x.tolist(), y[id_sort, :].tolist(),
                       linewidth=line_width,
                       color=plot_draw_styles[index_sort.numel() - id - 1]['color'],
                       linestyle=plot_draw_styles[index_sort.numel() - id - 1]['line_style'])

        plotted_lines.append(line[0])

        tracker = trackers[id_sort]
        disp_name = get_tracker_display_name(tracker)

        legend_text.append('{} [{:.1f}]'.format(disp_name, scores[id_sort]))

    try:
        ax.legend(plotted_lines[::-1], legend_text[::-1], loc=legend_loc, fancybox=False, edgecolor='black',
                  fontsize=font_size_legend, framealpha=1.0)
    except:
        pass

    ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim, title=title)
    ax.grid(True, linestyle='-.')
    fig.tight_layout()

    try:
        tikzplotlib.save('{}/{}_plot.tex'.format(result_plot_path, plot_type))
    except:
        pass
    fig.savefig('{}/{}_plot.pdf'.format(result_plot_path, plot_type), dpi=300, format='pdf', transparent=True)
    plt.draw()


# ==============================================================================
# ========================= 多机融合评估与安全校验逻辑 ============================
# ==============================================================================

def fuse_calc_seq_err_robust(pred_bb_a, pred_bb_b, anno_bb_a, anno_bb_b, dataset, target_visible_a, target_visible_b, score_a, score_b):
    pred_bb_a = pred_bb_a.clone()
    pred_bb_b = pred_bb_b.clone()

    pred_bb_a[0, :] = anno_bb_a[0, :]
    pred_bb_b[0, :] = anno_bb_b[0, :]

    fused_index = []
    for i, (s_a, s_b) in enumerate(zip(score_a, score_b)):
        score_list = [s_a, s_b]
        fused_index.append(score_list.index(max(score_list)))

    fused_index = torch.tensor(fused_index, dtype=torch.int64)
    min_len = min(len(fused_index), len(pred_bb_a))
    
    all_pred_bbox = torch.stack((pred_bb_a[:min_len], pred_bb_b[:min_len]))
    all_anno_bbox = torch.stack((anno_bb_a[:min_len], anno_bb_b[:min_len]))
    
    pred_bb = pred_bb_a[:min_len].clone()
    anno_bb = anno_bb_a[:min_len].clone()
    
    for i in range(min_len):
        pred_bb[i] = all_pred_bbox[fused_index[i]][i]
        anno_bb[i] = all_anno_bbox[fused_index[i]][i]

    # 💡 免疫 NoneType 错误的安全处理
    if target_visible_a is not None and target_visible_b is not None:
        target_visible = torch.ones(min_len, dtype=torch.bool)
        for i in range(min_len):
            target_visible[i] = target_visible_a[i] if fused_index[i] == 0 else target_visible_b[i]
        valid = ((anno_bb[:, 2:] > 0.0).sum(1) == 2) & target_visible
    else:
        target_visible = None
        valid = ((anno_bb[:, 2:] > 0.0).sum(1) == 2)

    err_center = calc_err_center(pred_bb, anno_bb)
    err_center_normalized = calc_err_center(pred_bb, anno_bb, normalized=True)
    err_overlap = calc_iou_overlap(pred_bb, anno_bb)

    if dataset in ['uav']:
        err_center[~valid] = -1.0
    else:
        err_center[~valid] = float("Inf")
    err_center_normalized[~valid] = -1.0
    err_overlap[~valid] = -1.0

    if dataset == 'lasot' and target_visible is not None:
        err_center_normalized[~target_visible] = float("Inf")
        err_center[~target_visible] = float("Inf")

    if torch.isnan(err_overlap).any():
        raise Exception('Nans in calculated overlap')
    return err_overlap, err_center, err_center_normalized, valid


def fuse_calc_seq_err_robust_APEC(pred_bb_a, pred_bb_b, anno_bb_a, anno_bb_b, dataset, target_visible_a, target_visible_b, score_a, score_b, APEC_a, APEC_b):
    pred_bb_a = pred_bb_a.clone()
    pred_bb_b = pred_bb_b.clone()

    pred_bb_a[0, :] = anno_bb_a[0, :]
    pred_bb_b[0, :] = anno_bb_b[0, :]

    fused_index = []
    for i, (s_a, s_b, A_a, A_b) in enumerate(zip(score_a, score_b, APEC_a, APEC_b)):
        if i > 0:
            q = 29
            avg_a = np.average(score_a[max(0,i-q):i+1])
            var_a = np.var(score_a[max(0,i-q):i+1])
            Avg_res_a = (avg_a - var_a)*APEC_a[i]
            
            avg_b = np.average(score_b[max(0,i-q):i+1])
            var_b = np.var(score_b[max(0,i-q):i+1])
            Avg_res_b = (avg_b - var_b)*APEC_b[i]

            Avg_res_list = [Avg_res_a, Avg_res_b]
            fused_index.append(Avg_res_list.index(max(Avg_res_list)))
        else:
            if (s_a < 0) and (s_b < 0):
                fused_index.append(0 if A_a >= A_b else 1)
            else:
                fused_index.append(0 if s_a >= s_b else 1)

    fused_index = torch.tensor(fused_index, dtype=torch.int64)
    min_len = min(len(fused_index), len(pred_bb_a))
    
    all_pred_bbox = torch.stack((pred_bb_a[:min_len], pred_bb_b[:min_len]))
    all_anno_bbox = torch.stack((anno_bb_a[:min_len], anno_bb_b[:min_len]))
    
    pred_bb = pred_bb_a[:min_len].clone()
    anno_bb = anno_bb_a[:min_len].clone()
    
    for i in range(min_len):
        pred_bb[i] = all_pred_bbox[fused_index[i]][i]
        anno_bb[i] = all_anno_bbox[fused_index[i]][i]

    if target_visible_a is not None and target_visible_b is not None:
        target_visible = torch.ones(min_len, dtype=torch.bool)
        for i in range(min_len):
            target_visible[i] = target_visible_a[i] if fused_index[i] == 0 else target_visible_b[i]
        valid = ((anno_bb[:, 2:] > 0.0).sum(1) == 2) & target_visible
    else:
        target_visible = None
        valid = ((anno_bb[:, 2:] > 0.0).sum(1) == 2)

    err_center = calc_err_center(pred_bb, anno_bb)
    err_center_normalized = calc_err_center(pred_bb, anno_bb, normalized=True)
    err_overlap = calc_iou_overlap(pred_bb, anno_bb)

    if dataset in ['uav']:
        err_center[~valid] = -1.0
    else:
        err_center[~valid] = float("Inf")
    err_center_normalized[~valid] = -1.0
    err_overlap[~valid] = -1.0

    if dataset == 'lasot' and target_visible is not None:
        err_center_normalized[~target_visible] = float("Inf")
        err_center[~target_visible] = float("Inf")

    if torch.isnan(err_overlap).any():
        raise Exception('Nans in calculated overlap')
    return err_overlap, err_center, err_center_normalized, valid

def fuse_extract_results(trackers, dataset, report_name, skip_missing_seq=False, plot_bin_gap=0.05, exclude_invalid_frames=False):
    settings = env_settings()
    result_plot_path = os.path.join(settings.result_plot_path, report_name)
    if not os.path.exists(result_plot_path):
        os.makedirs(result_plot_path)

    fuse_len = int(len(dataset)/2)
    threshold_set_overlap = torch.arange(0.0, 1.0 + plot_bin_gap, plot_bin_gap, dtype=torch.float64)
    threshold_set_center = torch.arange(0, 51, dtype=torch.float64)
    threshold_set_center_norm = torch.arange(0, 51, dtype=torch.float64) / 100.0

    num_variants = 3 # A, B, Fused
    avg_overlap_all = torch.zeros((fuse_len, len(trackers) * num_variants), dtype=torch.float64)
    ave_success_rate_plot_overlap = torch.zeros((fuse_len, len(trackers) * num_variants, threshold_set_overlap.numel()), dtype=torch.float32)
    ave_success_rate_plot_center = torch.zeros((fuse_len, len(trackers) * num_variants, threshold_set_center.numel()), dtype=torch.float32)
    ave_success_rate_plot_center_norm = torch.zeros((fuse_len, len(trackers) * num_variants, threshold_set_center.numel()), dtype=torch.float32)

    valid_sequence = torch.ones(fuse_len, dtype=torch.uint8)

    dataset_A = dataset[:fuse_len]
    dataset_B = dataset[fuse_len:]

    for seq_id, seq in enumerate(tqdm(dataset_A)):
        anno_bb_a = torch.tensor(seq.ground_truth_rect)
        anno_bb_b = torch.tensor(dataset_B[seq_id].ground_truth_rect)

        target_visible_a = torch.tensor(seq.target_visible, dtype=torch.uint8) if getattr(seq, 'target_visible', None) is not None else None
        target_visible_b = torch.tensor(dataset_B[seq_id].target_visible, dtype=torch.uint8) if getattr(dataset_B[seq_id], 'target_visible', None) is not None else None

        for trk_id, trk in enumerate(trackers):
            base_results_path_a = '{}/{}'.format(trk.results_dir, seq.name)
            base_results_path_b = '{}/{}'.format(trk.results_dir, dataset_B[seq_id].name)
            results_path_a = '{}.txt'.format(base_results_path_a)
            results_path_b = '{}.txt'.format(base_results_path_b)
            score_path_a = '{}_max_score.txt'.format(base_results_path_a)
            score_path_b = '{}_max_score.txt'.format(base_results_path_b)
            APCE_path_a = '{}_APCE.txt'.format(base_results_path_a)
            APCE_path_b = '{}_APCE.txt'.format(base_results_path_b)

            if os.path.isfile(results_path_a) and os.path.isfile(results_path_b):
                pred_bb_a = torch.tensor(load_text(str(results_path_a), delimiter=('\t', ','), dtype=np.float64))
                pred_bb_b = torch.tensor(load_text(str(results_path_b), delimiter=('\t', ','), dtype=np.float64))
                score_a = np.loadtxt(str(score_path_a), dtype=np.float64)
                score_b = np.loadtxt(str(score_path_b), dtype=np.float64)
            else:
                if skip_missing_seq:
                    valid_sequence[seq_id] = 0
                    break
                else:
                    raise Exception('Result not found. {}'.format(results_path_a))

            # 单机评测
            err_ov_a, err_ct_a, err_ctn_a, val_a = calc_seq_err_robust(pred_bb_a, anno_bb_a, seq.dataset, target_visible_a)
            err_ov_b, err_ct_b, err_ctn_b, val_b = calc_seq_err_robust(pred_bb_b, anno_bb_b, seq.dataset, target_visible_b)

            # 融合评测
            if os.path.isfile(APCE_path_a):
                APEC_a = np.loadtxt(str(APCE_path_a), dtype=np.float64)
                APEC_b = np.loadtxt(str(APCE_path_b), dtype=np.float64)
                err_ov_f, err_ct_f, err_ctn_f, val_f = fuse_calc_seq_err_robust_APEC(
                    pred_bb_a, pred_bb_b, anno_bb_a, anno_bb_b, seq.dataset, 
                    target_visible_a, target_visible_b, score_a, score_b, APEC_a, APEC_b)
            else:
                err_ov_f, err_ct_f, err_ctn_f, val_f = fuse_calc_seq_err_robust(
                    pred_bb_a, pred_bb_b, anno_bb_a, anno_bb_b, seq.dataset, 
                    target_visible_a, target_visible_b, score_a, score_b)

            base_idx = trk_id * num_variants
            def store_variant(offset, ov, ct, ctn, val_f, shape):
                avg_overlap_all[seq_id, base_idx + offset] = ov[val_f].mean()
                slen = val_f.long().sum() if exclude_invalid_frames else shape[0]
                if slen <= 0: raise Exception('Seq length zero')
                ave_success_rate_plot_overlap[seq_id, base_idx + offset, :] = (ov.view(-1, 1) > threshold_set_overlap.view(1, -1)).sum(0).float() / slen
                ave_success_rate_plot_center[seq_id, base_idx + offset, :] = (ct.view(-1, 1) <= threshold_set_center.view(1, -1)).sum(0).float() / slen
                ave_success_rate_plot_center_norm[seq_id, base_idx + offset, :] = (ctn.view(-1, 1) <= threshold_set_center_norm.view(1, -1)).sum(0).float() / slen

            store_variant(0, err_ov_a, err_ct_a, err_ctn_a, val_a, anno_bb_a.shape)
            store_variant(1, err_ov_b, err_ct_b, err_ctn_b, val_b, anno_bb_b.shape)
            store_variant(2, err_ov_f, err_ct_f, err_ctn_f, val_f, anno_bb_a.shape)

    print('\n\nComputed results over {} / {} sequences'.format(valid_sequence.long().sum().item(), valid_sequence.shape[0]))

    seq_names = [s.name for s in dataset_A]
    tracker_names = []
    for t in trackers:
        base_disp_name = t.display_name if t.display_name else t.name
        tracker_names.append({'name': t.name, 'param': t.parameter_name, 'run_id': t.run_id, 'disp_name': base_disp_name + ' (Drone A)'})
        tracker_names.append({'name': t.name, 'param': t.parameter_name, 'run_id': t.run_id, 'disp_name': base_disp_name + ' (Drone B)'})
        tracker_names.append({'name': t.name, 'param': t.parameter_name, 'run_id': t.run_id, 'disp_name': base_disp_name + ' (Fused)'})

    eval_data = {'sequences': seq_names, 'trackers': tracker_names,
                 'valid_sequence': valid_sequence.tolist(),
                 'ave_success_rate_plot_overlap': ave_success_rate_plot_overlap.tolist(),
                 'ave_success_rate_plot_center': ave_success_rate_plot_center.tolist(),
                 'ave_success_rate_plot_center_norm': ave_success_rate_plot_center_norm.tolist(),
                 'avg_overlap_all': avg_overlap_all.tolist(),
                 'threshold_set_overlap': threshold_set_overlap.tolist(),
                 'threshold_set_center': threshold_set_center.tolist(),
                 'threshold_set_center_norm': threshold_set_center_norm.tolist()}

    with open(result_plot_path + '/eval_data.pkl', 'wb') as fh:
        pickle.dump(eval_data, fh)

    return eval_data


def three_fuse_calc_seq_err_robust(pred_bb_a, pred_bb_b, pred_bb_c, anno_bb_a, anno_bb_b, anno_bb_c, dataset, target_visible_a, target_visible_b, target_visible_c, score_a, score_b, score_c):
    pred_bb_a = pred_bb_a.clone()
    pred_bb_b = pred_bb_b.clone()
    pred_bb_c = pred_bb_c.clone()

    pred_bb_a[0, :] = anno_bb_a[0, :]
    pred_bb_b[0, :] = anno_bb_b[0, :]
    pred_bb_c[0, :] = anno_bb_c[0, :]

    fused_index = []
    for i, (s_a, s_b, s_c) in enumerate(zip(score_a, score_b, score_c)):
        score_list = [s_a, s_b, s_c]
        fused_index.append(score_list.index(max(score_list)))

    fused_index = torch.tensor(fused_index, dtype=torch.int64)
    min_len = min(len(fused_index), len(pred_bb_a))

    all_pred_bbox = torch.stack((pred_bb_a[:min_len], pred_bb_b[:min_len], pred_bb_c[:min_len]))
    all_anno_bbox = torch.stack((anno_bb_a[:min_len], anno_bb_b[:min_len], anno_bb_c[:min_len]))
    
    pred_bb = pred_bb_a[:min_len].clone()
    anno_bb = anno_bb_a[:min_len].clone()
    
    for i in range(min_len):
        pred_bb[i] = all_pred_bbox[fused_index[i]][i]
        anno_bb[i] = all_anno_bbox[fused_index[i]][i]

    if target_visible_a is not None and target_visible_b is not None and target_visible_c is not None:
        target_visible = torch.ones(min_len, dtype=torch.bool)
        for i in range(min_len):
            if fused_index[i] == 0: target_visible[i] = target_visible_a[i]
            elif fused_index[i] == 1: target_visible[i] = target_visible_b[i]
            else: target_visible[i] = target_visible_c[i]
        valid = ((anno_bb[:, 2:] > 0.0).sum(1) == 2) & target_visible
    else:
        target_visible = None
        valid = ((anno_bb[:, 2:] > 0.0).sum(1) == 2)

    err_center = calc_err_center(pred_bb, anno_bb)
    err_center_normalized = calc_err_center(pred_bb, anno_bb, normalized=True)
    err_overlap = calc_iou_overlap(pred_bb, anno_bb)

    if dataset in ['uav']:
        err_center[~valid] = -1.0
    else:
        err_center[~valid] = float("Inf")
    err_center_normalized[~valid] = -1.0
    err_overlap[~valid] = -1.0

    if dataset == 'lasot' and target_visible is not None:
        err_center_normalized[~target_visible] = float("Inf")
        err_center[~target_visible] = float("Inf")

    if torch.isnan(err_overlap).any():
        raise Exception('Nans in calculated overlap')
    return err_overlap, err_center, err_center_normalized, valid


def three_fuse_calc_seq_err_robust_APEC(pred_bb_a, pred_bb_b, pred_bb_c, anno_bb_a, anno_bb_b, anno_bb_c, dataset, target_visible_a, target_visible_b, target_visible_c, score_a, score_b, score_c, APEC_a, APEC_b, APEC_c):
    pred_bb_a = pred_bb_a.clone()
    pred_bb_b = pred_bb_b.clone()
    pred_bb_c = pred_bb_c.clone()

    pred_bb_a[0, :] = anno_bb_a[0, :]
    pred_bb_b[0, :] = anno_bb_b[0, :]
    pred_bb_c[0, :] = anno_bb_c[0, :]

    fused_index = []
    for i, (s_a, s_b, s_c, A_a, A_b, A_c) in enumerate(zip(score_a, score_b, score_c, APEC_a, APEC_b, APEC_c)):
        if i > 0:
            q = 44   
            avg_a = np.average(score_a[max(0,i-q):i+1])
            var_a = np.var(score_a[max(0,i-q):i+1])
            Avg_res_a = (avg_a - var_a)*APEC_a[i]
            
            avg_b = np.average(score_b[max(0,i-q):i+1])
            var_b = np.var(score_b[max(0,i-q):i+1])
            Avg_res_b = (avg_b - var_b)*APEC_b[i]
            
            avg_c = np.average(score_c[max(0,i-q):i+1])
            var_c = np.var(score_c[max(0,i-q):i+1])
            Avg_res_c = (avg_c - var_c)*APEC_c[i]

            Avg_res_list = [Avg_res_a, Avg_res_b, Avg_res_c]
            fused_index.append(Avg_res_list.index(max(Avg_res_list)))
        else:
            score_list = [s_a, s_b, s_c]
            Apec_list = [A_a, A_b, A_c]
            if max(score_list) < 0:
                fused_index.append(score_list.index(max(score_list)))
            else:
                fused_index.append(Apec_list.index(max(Apec_list)))

    fused_index = torch.tensor(fused_index, dtype=torch.int64)
    min_len = min(len(fused_index), len(pred_bb_a))

    all_pred_bbox = torch.stack((pred_bb_a[:min_len], pred_bb_b[:min_len], pred_bb_c[:min_len]))
    all_anno_bbox = torch.stack((anno_bb_a[:min_len], anno_bb_b[:min_len], anno_bb_c[:min_len]))
    
    pred_bb = pred_bb_a[:min_len].clone()
    anno_bb = anno_bb_a[:min_len].clone()
    
    for i in range(min_len):
        pred_bb[i] = all_pred_bbox[fused_index[i]][i]
        anno_bb[i] = all_anno_bbox[fused_index[i]][i]

    if target_visible_a is not None and target_visible_b is not None and target_visible_c is not None:
        target_visible = torch.ones(min_len, dtype=torch.bool)
        for i in range(min_len):
            if fused_index[i] == 0: target_visible[i] = target_visible_a[i]
            elif fused_index[i] == 1: target_visible[i] = target_visible_b[i]
            else: target_visible[i] = target_visible_c[i]
        valid = ((anno_bb[:, 2:] > 0.0).sum(1) == 2) & target_visible
    else:
        target_visible = None
        valid = ((anno_bb[:, 2:] > 0.0).sum(1) == 2)

    err_center = calc_err_center(pred_bb, anno_bb)
    err_center_normalized = calc_err_center(pred_bb, anno_bb, normalized=True)
    err_overlap = calc_iou_overlap(pred_bb, anno_bb)

    if dataset in ['uav']:
        err_center[~valid] = -1.0
    else:
        err_center[~valid] = float("Inf")
    err_center_normalized[~valid] = -1.0
    err_overlap[~valid] = -1.0

    if dataset == 'lasot' and target_visible is not None:
        err_center_normalized[~target_visible] = float("Inf")
        err_center[~target_visible] = float("Inf")

    if torch.isnan(err_overlap).any():
        raise Exception('Nans in calculated overlap')
    return err_overlap, err_center, err_center_normalized, valid


# 💡 核心修改：三机抽取与 4倍虚拟 Tracker 生成
def three_fuse_extract_results_APCE(trackers, dataset, report_name, skip_missing_seq=False, plot_bin_gap=0.05, exclude_invalid_frames=False):
    settings = env_settings()
    
    result_plot_path = os.path.join(settings.result_plot_path, report_name)
    if not os.path.exists(result_plot_path):
        os.makedirs(result_plot_path)

    fuse_len = int(len(dataset)/3)

    threshold_set_overlap = torch.arange(0.0, 1.0 + plot_bin_gap, plot_bin_gap, dtype=torch.float64)
    threshold_set_center = torch.arange(0, 51, dtype=torch.float64)
    threshold_set_center_norm = torch.arange(0, 51, dtype=torch.float64) / 100.0

    num_variants = 4 # A机, B机, C机, 融合结果
    avg_overlap_all = torch.zeros((fuse_len, len(trackers) * num_variants), dtype=torch.float64)
    ave_success_rate_plot_overlap = torch.zeros((fuse_len, len(trackers) * num_variants, threshold_set_overlap.numel()), dtype=torch.float32)
    ave_success_rate_plot_center = torch.zeros((fuse_len, len(trackers) * num_variants, threshold_set_center.numel()), dtype=torch.float32)
    ave_success_rate_plot_center_norm = torch.zeros((fuse_len, len(trackers) * num_variants, threshold_set_center.numel()), dtype=torch.float32)

    valid_sequence = torch.ones(fuse_len, dtype=torch.uint8)

    dataset_A = dataset[:fuse_len]
    dataset_B = dataset[fuse_len:(2*fuse_len)]
    dataset_C = dataset[(2*fuse_len):]

    for seq_id, seq in enumerate(tqdm(dataset_A)):
        anno_bb_a = torch.tensor(seq.ground_truth_rect)
        anno_bb_b = torch.tensor(dataset_B[seq_id].ground_truth_rect)
        anno_bb_c = torch.tensor(dataset_C[seq_id].ground_truth_rect)

        target_visible_a = torch.tensor(seq.target_visible, dtype=torch.uint8) if getattr(seq, 'target_visible', None) is not None else None
        target_visible_b = torch.tensor(dataset_B[seq_id].target_visible, dtype=torch.uint8) if getattr(dataset_B[seq_id], 'target_visible', None) is not None else None
        target_visible_c = torch.tensor(dataset_C[seq_id].target_visible, dtype=torch.uint8) if getattr(dataset_C[seq_id], 'target_visible', None) is not None else None

        for trk_id, trk in enumerate(trackers):
            base_results_path_a = '{}/{}'.format(trk.results_dir, seq.name)
            base_results_path_b = '{}/{}'.format(trk.results_dir, dataset_B[seq_id].name)
            base_results_path_c = '{}/{}'.format(trk.results_dir, dataset_C[seq_id].name)

            results_path_a = '{}.txt'.format(base_results_path_a)
            results_path_b = '{}.txt'.format(base_results_path_b)
            results_path_c = '{}.txt'.format(base_results_path_c)

            score_path_a = '{}_max_score.txt'.format(base_results_path_a)
            score_path_b = '{}_max_score.txt'.format(base_results_path_b)
            score_path_c = '{}_max_score.txt'.format(base_results_path_c)

            APCE_path_a = '{}_APCE.txt'.format(base_results_path_a)
            APCE_path_b = '{}_APCE.txt'.format(base_results_path_b)
            APCE_path_c = '{}_APCE.txt'.format(base_results_path_c)

            if os.path.isfile(results_path_a) and os.path.isfile(results_path_b) and os.path.isfile(results_path_c):
                pred_bb_a = torch.tensor(load_text(str(results_path_a), delimiter=('\t', ','), dtype=np.float64))
                pred_bb_b = torch.tensor(load_text(str(results_path_b), delimiter=('\t', ','), dtype=np.float64))
                pred_bb_c = torch.tensor(load_text(str(results_path_c), delimiter=('\t', ','), dtype=np.float64))
                score_a = np.loadtxt(str(score_path_a), dtype=np.float64)
                score_b = np.loadtxt(str(score_path_b), dtype=np.float64)
                score_c = np.loadtxt(str(score_path_c), dtype=np.float64)
            else:
                if skip_missing_seq:
                    valid_sequence[seq_id] = 0
                    break
                else:
                    raise Exception('Result not found. {}'.format(results_path_a))

            # 分别计算单机误差
            err_ov_a, err_ct_a, err_ctn_a, val_a = calc_seq_err_robust(pred_bb_a, anno_bb_a, seq.dataset, target_visible_a)
            err_ov_b, err_ct_b, err_ctn_b, val_b = calc_seq_err_robust(pred_bb_b, anno_bb_b, seq.dataset, target_visible_b)
            err_ov_c, err_ct_c, err_ctn_c, val_c = calc_seq_err_robust(pred_bb_c, anno_bb_c, seq.dataset, target_visible_c)

            # 计算融合误差
            if os.path.isfile(APCE_path_a):
                APEC_a = np.loadtxt(str(APCE_path_a), dtype=np.float64)
                APEC_b = np.loadtxt(str(APCE_path_b), dtype=np.float64)
                APEC_c = np.loadtxt(str(APCE_path_c), dtype=np.float64)

                err_ov_f, err_ct_f, err_ctn_f, val_f = three_fuse_calc_seq_err_robust_APEC(
                    pred_bb_a, pred_bb_b, pred_bb_c, anno_bb_a, anno_bb_b, anno_bb_c, seq.dataset, 
                    target_visible_a, target_visible_b, target_visible_c, 
                    score_a, score_b, score_c, APEC_a, APEC_b, APEC_c)
            else:
                err_ov_f, err_ct_f, err_ctn_f, val_f = three_fuse_calc_seq_err_robust(
                    pred_bb_a, pred_bb_b, pred_bb_c, anno_bb_a, anno_bb_b, anno_bb_c, seq.dataset, 
                    target_visible_a, target_visible_b, target_visible_c, 
                    score_a, score_b, score_c)

            base_idx = trk_id * num_variants
            
            def store_variant_result(offset_idx, err_overlap, err_center, err_center_normalized, valid_frame, anno_bb_shape):
                avg_overlap_all[seq_id, base_idx + offset_idx] = err_overlap[valid_frame].mean()
                seq_length = valid_frame.long().sum() if exclude_invalid_frames else anno_bb_shape[0]
                if seq_length <= 0: raise Exception('Seq length zero')

                ave_success_rate_plot_overlap[seq_id, base_idx + offset_idx, :] = (err_overlap.view(-1, 1) > threshold_set_overlap.view(1, -1)).sum(0).float() / seq_length
                ave_success_rate_plot_center[seq_id, base_idx + offset_idx, :] = (err_center.view(-1, 1) <= threshold_set_center.view(1, -1)).sum(0).float() / seq_length
                ave_success_rate_plot_center_norm[seq_id, base_idx + offset_idx, :] = (err_center_normalized.view(-1, 1) <= threshold_set_center_norm.view(1, -1)).sum(0).float() / seq_length

            store_variant_result(0, err_ov_a, err_ct_a, err_ctn_a, val_a, anno_bb_a.shape)
            store_variant_result(1, err_ov_b, err_ct_b, err_ctn_b, val_b, anno_bb_b.shape)
            store_variant_result(2, err_ov_c, err_ct_c, err_ctn_c, val_c, anno_bb_c.shape)
            store_variant_result(3, err_ov_f, err_ct_f, err_ctn_f, val_f, anno_bb_a.shape)

    print('\n\nComputed results over {} / {} sequences'.format(valid_sequence.long().sum().item(), valid_sequence.shape[0]))

    seq_names = [s.name for s in dataset_A]
    
    # 将 Tracker 名字也裂变成 4 个，分别显示
    tracker_names = []
    for t in trackers:
        base_disp_name = t.display_name if t.display_name else t.name
        tracker_names.append({'name': t.name, 'param': t.parameter_name, 'run_id': t.run_id, 'disp_name': base_disp_name + ' (Drone A)'})
        tracker_names.append({'name': t.name, 'param': t.parameter_name, 'run_id': t.run_id, 'disp_name': base_disp_name + ' (Drone B)'})
        tracker_names.append({'name': t.name, 'param': t.parameter_name, 'run_id': t.run_id, 'disp_name': base_disp_name + ' (Drone C)'})
        tracker_names.append({'name': t.name, 'param': t.parameter_name, 'run_id': t.run_id, 'disp_name': base_disp_name + ' (Fused)'})

    eval_data = {'sequences': seq_names, 'trackers': tracker_names,
                 'valid_sequence': valid_sequence.tolist(),
                 'ave_success_rate_plot_overlap': ave_success_rate_plot_overlap.tolist(),
                 'ave_success_rate_plot_center': ave_success_rate_plot_center.tolist(),
                 'ave_success_rate_plot_center_norm': ave_success_rate_plot_center_norm.tolist(),
                 'avg_overlap_all': avg_overlap_all.tolist(),
                 'threshold_set_overlap': threshold_set_overlap.tolist(),
                 'threshold_set_center': threshold_set_center.tolist(),
                 'threshold_set_center_norm': threshold_set_center_norm.tolist()}

    with open(result_plot_path + '/eval_data.pkl', 'wb') as fh:
        pickle.dump(eval_data, fh)

    return eval_data
# 💡 核心修改：三机抽取与 4倍虚拟 Tracker 生成 (纯净版 AFS 融合，无 APCE)
def three_fuse_extract_results(trackers, dataset, report_name, skip_missing_seq=False, plot_bin_gap=0.05, exclude_invalid_frames=False):
    settings = env_settings()
    
    result_plot_path = os.path.join(settings.result_plot_path, report_name)
    if not os.path.exists(result_plot_path):
        os.makedirs(result_plot_path)

    fuse_len = int(len(dataset)/3)

    threshold_set_overlap = torch.arange(0.0, 1.0 + plot_bin_gap, plot_bin_gap, dtype=torch.float64)
    threshold_set_center = torch.arange(0, 51, dtype=torch.float64)
    threshold_set_center_norm = torch.arange(0, 51, dtype=torch.float64) / 100.0

    num_variants = 4 # A机, B机, C机, 融合结果
    avg_overlap_all = torch.zeros((fuse_len, len(trackers) * num_variants), dtype=torch.float64)
    ave_success_rate_plot_overlap = torch.zeros((fuse_len, len(trackers) * num_variants, threshold_set_overlap.numel()), dtype=torch.float32)
    ave_success_rate_plot_center = torch.zeros((fuse_len, len(trackers) * num_variants, threshold_set_center.numel()), dtype=torch.float32)
    ave_success_rate_plot_center_norm = torch.zeros((fuse_len, len(trackers) * num_variants, threshold_set_center.numel()), dtype=torch.float32)

    valid_sequence = torch.ones(fuse_len, dtype=torch.uint8)

    dataset_A = dataset[:fuse_len]
    dataset_B = dataset[fuse_len:(2*fuse_len)]
    dataset_C = dataset[(2*fuse_len):]

    for seq_id, seq in enumerate(tqdm(dataset_A)):
        anno_bb_a = torch.tensor(seq.ground_truth_rect)
        anno_bb_b = torch.tensor(dataset_B[seq_id].ground_truth_rect)
        anno_bb_c = torch.tensor(dataset_C[seq_id].ground_truth_rect)

        target_visible_a = torch.tensor(seq.target_visible, dtype=torch.uint8) if getattr(seq, 'target_visible', None) is not None else None
        target_visible_b = torch.tensor(dataset_B[seq_id].target_visible, dtype=torch.uint8) if getattr(dataset_B[seq_id], 'target_visible', None) is not None else None
        target_visible_c = torch.tensor(dataset_C[seq_id].target_visible, dtype=torch.uint8) if getattr(dataset_C[seq_id], 'target_visible', None) is not None else None

        for trk_id, trk in enumerate(trackers):
            base_results_path_a = '{}/{}'.format(trk.results_dir, seq.name)
            base_results_path_b = '{}/{}'.format(trk.results_dir, dataset_B[seq_id].name)
            base_results_path_c = '{}/{}'.format(trk.results_dir, dataset_C[seq_id].name)

            results_path_a = '{}.txt'.format(base_results_path_a)
            results_path_b = '{}.txt'.format(base_results_path_b)
            results_path_c = '{}.txt'.format(base_results_path_c)

            score_path_a = '{}_max_score.txt'.format(base_results_path_a)
            score_path_b = '{}_max_score.txt'.format(base_results_path_b)
            score_path_c = '{}_max_score.txt'.format(base_results_path_c)

            # ✂️ 【修改 1】：删除了读取 APCE_path_x 的相关代码

            if os.path.isfile(results_path_a) and os.path.isfile(results_path_b) and os.path.isfile(results_path_c):
                pred_bb_a = torch.tensor(load_text(str(results_path_a), delimiter=('\t', ','), dtype=np.float64))
                pred_bb_b = torch.tensor(load_text(str(results_path_b), delimiter=('\t', ','), dtype=np.float64))
                pred_bb_c = torch.tensor(load_text(str(results_path_c), delimiter=('\t', ','), dtype=np.float64))
                score_a = np.loadtxt(str(score_path_a), dtype=np.float64)
                score_b = np.loadtxt(str(score_path_b), dtype=np.float64)
                score_c = np.loadtxt(str(score_path_c), dtype=np.float64)
            else:
                if skip_missing_seq:
                    valid_sequence[seq_id] = 0
                    break
                else:
                    raise Exception('Result not found. {}'.format(results_path_a))

            # 分别计算单机误差
            err_ov_a, err_ct_a, err_ctn_a, val_a = calc_seq_err_robust(pred_bb_a, anno_bb_a, seq.dataset, target_visible_a)
            err_ov_b, err_ct_b, err_ctn_b, val_b = calc_seq_err_robust(pred_bb_b, anno_bb_b, seq.dataset, target_visible_b)
            err_ov_c, err_ct_c, err_ctn_c, val_c = calc_seq_err_robust(pred_bb_c, anno_bb_c, seq.dataset, target_visible_c)

            # ✂️ 【修改 2】：去掉了 if os.path.isfile(APCE_path_a) 分支。
            # 强制使用纯 AFS 逻辑 (仅比较 score_a, score_b, score_c)
            err_ov_f, err_ct_f, err_ctn_f, val_f = three_fuse_calc_seq_err_robust(
                pred_bb_a, pred_bb_b, pred_bb_c, anno_bb_a, anno_bb_b, anno_bb_c, seq.dataset, 
                target_visible_a, target_visible_b, target_visible_c, 
                score_a, score_b, score_c)

            base_idx = trk_id * num_variants
            
            def store_variant_result(offset_idx, err_overlap, err_center, err_center_normalized, valid_frame, anno_bb_shape):
                avg_overlap_all[seq_id, base_idx + offset_idx] = err_overlap[valid_frame].mean()
                seq_length = valid_frame.long().sum() if exclude_invalid_frames else anno_bb_shape[0]
                if seq_length <= 0: raise Exception('Seq length zero')

                ave_success_rate_plot_overlap[seq_id, base_idx + offset_idx, :] = (err_overlap.view(-1, 1) > threshold_set_overlap.view(1, -1)).sum(0).float() / seq_length
                ave_success_rate_plot_center[seq_id, base_idx + offset_idx, :] = (err_center.view(-1, 1) <= threshold_set_center.view(1, -1)).sum(0).float() / seq_length
                ave_success_rate_plot_center_norm[seq_id, base_idx + offset_idx, :] = (err_center_normalized.view(-1, 1) <= threshold_set_center_norm.view(1, -1)).sum(0).float() / seq_length

            store_variant_result(0, err_ov_a, err_ct_a, err_ctn_a, val_a, anno_bb_a.shape)
            store_variant_result(1, err_ov_b, err_ct_b, err_ctn_b, val_b, anno_bb_b.shape)
            store_variant_result(2, err_ov_c, err_ct_c, err_ctn_c, val_c, anno_bb_c.shape)
            store_variant_result(3, err_ov_f, err_ct_f, err_ctn_f, val_f, anno_bb_a.shape)

    print('\n\nComputed results over {} / {} sequences'.format(valid_sequence.long().sum().item(), valid_sequence.shape[0]))

    seq_names = [s.name for s in dataset_A]
    
    # 将 Tracker 名字也裂变成 4 个，分别显示
    tracker_names = []
    for t in trackers:
        base_disp_name = t.display_name if t.display_name else t.name
        tracker_names.append({'name': t.name, 'param': t.parameter_name, 'run_id': t.run_id, 'disp_name': base_disp_name + ' (Drone A)'})
        tracker_names.append({'name': t.name, 'param': t.parameter_name, 'run_id': t.run_id, 'disp_name': base_disp_name + ' (Drone B)'})
        tracker_names.append({'name': t.name, 'param': t.parameter_name, 'run_id': t.run_id, 'disp_name': base_disp_name + ' (Drone C)'})
        tracker_names.append({'name': t.name, 'param': t.parameter_name, 'run_id': t.run_id, 'disp_name': base_disp_name + ' (Fused)'})

    eval_data = {'sequences': seq_names, 'trackers': tracker_names,
                 'valid_sequence': valid_sequence.tolist(),
                 'ave_success_rate_plot_overlap': ave_success_rate_plot_overlap.tolist(),
                 'ave_success_rate_plot_center': ave_success_rate_plot_center.tolist(),
                 'ave_success_rate_plot_center_norm': ave_success_rate_plot_center_norm.tolist(),
                 'avg_overlap_all': avg_overlap_all.tolist(),
                 'threshold_set_overlap': threshold_set_overlap.tolist(),
                 'threshold_set_center': threshold_set_center.tolist(),
                 'threshold_set_center_norm': threshold_set_center_norm.tolist()}

    with open(result_plot_path + '/eval_data.pkl', 'wb') as fh:
        pickle.dump(eval_data, fh)

    return eval_data

def check_and_load_precomputed_results(trackers, dataset, report_name, force_evaluation=False, **kwargs):
    settings = env_settings()
    result_plot_path = os.path.join(settings.result_plot_path, report_name)
    eval_data_path = os.path.join(result_plot_path, 'eval_data.pkl')

    # 💡 根据数据集长度自动判断需要走哪一种评估逻辑
    dataset_len = len(dataset)
    
    if dataset_len > 0 and dataset_len % 3 == 0 and 'three' in report_name.lower():
        print(">>> Triggering THREE-DRONE fusion evaluation...")
        #eval_data = three_fuse_extract_results(trackers, dataset, report_name, **kwargs)
        eval_data = three_fuse_extract_results_APCE(trackers, dataset, report_name, **kwargs)
    elif dataset_len > 0 and dataset_len % 2 == 0 and 'fuse' in report_name.lower():
        print(">>> Triggering TWO-DRONE fusion evaluation...")
        eval_data = fuse_extract_results(trackers, dataset, report_name, **kwargs) 
    else:
        print(">>> Triggering SINGLE-DRONE standard evaluation...")
        eval_data = extract_results(trackers, dataset, report_name, **kwargs)

    with open(eval_data_path, 'wb') as fh:
        pickle.dump(eval_data, fh)
    return eval_data


def get_auc_curve(ave_success_rate_plot_overlap, valid_sequence):
    ave_success_rate_plot_overlap = ave_success_rate_plot_overlap[valid_sequence, :, :]
    auc_curve = ave_success_rate_plot_overlap.mean(0) * 100.0
    auc = auc_curve.mean(-1)
    return auc_curve, auc

def get_prec_curve(ave_success_rate_plot_center, valid_sequence):
    ave_success_rate_plot_center = ave_success_rate_plot_center[valid_sequence, :, :]
    prec_curve = ave_success_rate_plot_center.mean(0) * 100.0
    prec_score = prec_curve[:, 20]
    return prec_curve, prec_score

def plot_results(trackers, dataset, report_name, merge_results=False,
                 plot_types=('success'), force_evaluation=False, **kwargs):
    settings = env_settings()
    plot_draw_styles = get_plot_draw_styles()

    result_plot_path = os.path.join(settings.result_plot_path, report_name)
    eval_data = check_and_load_precomputed_results(trackers, dataset, report_name, force_evaluation, **kwargs)

    if merge_results:
        eval_data = merge_multiple_runs(eval_data)

    tracker_names = eval_data['trackers']
    valid_sequence = torch.tensor(eval_data['valid_sequence'], dtype=torch.bool)

    print('\nPlotting results over {} / {} sequences'.format(valid_sequence.long().sum().item(), valid_sequence.shape[0]))
    print('\nGenerating plots for: {}'.format(report_name))

    # ******************************** Success Plot **************************************
    if 'success' in plot_types:
        ave_success_rate_plot_overlap = torch.tensor(eval_data['ave_success_rate_plot_overlap'])
        auc_curve, auc = get_auc_curve(ave_success_rate_plot_overlap, valid_sequence)
        threshold_set_overlap = torch.tensor(eval_data['threshold_set_overlap'])

        success_plot_opts = {'plot_type': 'success', 'legend_loc': 'lower left', 'xlabel': 'Overlap threshold',
                             'ylabel': 'Overlap Precision [%]', 'xlim': (0, 1.0), 'ylim': (0, 88), 'title': 'Success'}
        plot_draw_save(auc_curve, threshold_set_overlap, auc, tracker_names, plot_draw_styles, result_plot_path, success_plot_opts)

    # ******************************** Precision Plot **************************************
    if 'prec' in plot_types:
        ave_success_rate_plot_center = torch.tensor(eval_data['ave_success_rate_plot_center'])
        prec_curve, prec_score = get_prec_curve(ave_success_rate_plot_center, valid_sequence)
        threshold_set_center = torch.tensor(eval_data['threshold_set_center'])

        precision_plot_opts = {'plot_type': 'precision', 'legend_loc': 'lower right',
                               'xlabel': 'Location error threshold [pixels]', 'ylabel': 'Distance Precision [%]',
                               'xlim': (0, 50), 'ylim': (0, 100), 'title': 'Precision plot'}
        plot_draw_save(prec_curve, threshold_set_center, prec_score, tracker_names, plot_draw_styles, result_plot_path,
                       precision_plot_opts)

    # ******************************** Norm Precision Plot **************************************
    if 'norm_prec' in plot_types:
        ave_success_rate_plot_center_norm = torch.tensor(eval_data['ave_success_rate_plot_center_norm'])
        prec_curve, prec_score = get_prec_curve(ave_success_rate_plot_center_norm, valid_sequence)
        threshold_set_center_norm = torch.tensor(eval_data['threshold_set_center_norm'])

        norm_precision_plot_opts = {'plot_type': 'norm_precision', 'legend_loc': 'lower right',
                                    'xlabel': 'Location error threshold', 'ylabel': 'Distance Precision [%]',
                                    'xlim': (0, 0.5), 'ylim': (0, 85), 'title': 'Normalized Precision'}
        plot_draw_save(prec_curve, threshold_set_center_norm, prec_score, tracker_names, plot_draw_styles, result_plot_path,
                       norm_precision_plot_opts)

    plt.show()


def generate_formatted_report(row_labels, scores, table_name=''):
    name_width = max([len(d) for d in row_labels] + [len(table_name)]) + 5
    min_score_width = 10

    report_text = '\n{label: <{width}} |'.format(label=table_name, width=name_width)
    score_widths = [max(min_score_width, len(k) + 3) for k in scores.keys()]

    for s, s_w in zip(scores.keys(), score_widths):
        report_text = '{prev} {s: <{width}} |'.format(prev=report_text, s=s, width=s_w)
    report_text = '{prev}\n'.format(prev=report_text)

    for trk_id, d_name in enumerate(row_labels):
        report_text = '{prev}{tracker: <{width}} |'.format(prev=report_text, tracker=d_name, width=name_width)
        for (score_type, score_value), s_w in zip(scores.items(), score_widths):
            report_text = '{prev} {score: <{width}} |'.format(prev=report_text,
                                                              score='{:0.2f}'.format(score_value[trk_id].item()),
                                                              width=s_w)
        report_text = '{prev}\n'.format(prev=report_text)
    return report_text


def print_results(trackers, dataset, report_name, merge_results=False,
                  plot_types=('success'), **kwargs):
    eval_data = check_and_load_precomputed_results(trackers, dataset, report_name, **kwargs)

    if merge_results:
        eval_data = merge_multiple_runs(eval_data)

    tracker_names = eval_data['trackers']
    valid_sequence = torch.tensor(eval_data['valid_sequence'], dtype=torch.bool)

    print('\nReporting results over {} / {} sequences'.format(valid_sequence.long().sum().item(), valid_sequence.shape[0]))

    scores = {}
    if 'success' in plot_types:
        threshold_set_overlap = torch.tensor(eval_data['threshold_set_overlap'])
        ave_success_rate_plot_overlap = torch.tensor(eval_data['ave_success_rate_plot_overlap'])
        auc_curve, auc = get_auc_curve(ave_success_rate_plot_overlap, valid_sequence)
        scores['AUC'] = auc
        scores['OP50'] = auc_curve[:, threshold_set_overlap == 0.50]
        scores['OP75'] = auc_curve[:, threshold_set_overlap == 0.75]

    if 'prec' in plot_types:
        ave_success_rate_plot_center = torch.tensor(eval_data['ave_success_rate_plot_center'])
        prec_curve, prec_score = get_prec_curve(ave_success_rate_plot_center, valid_sequence)
        scores['Precision'] = prec_score

    if 'norm_prec' in plot_types:
        ave_success_rate_plot_center_norm = torch.tensor(eval_data['ave_success_rate_plot_center_norm'])
        norm_prec_curve, norm_prec_score = get_prec_curve(ave_success_rate_plot_center_norm, valid_sequence)
        scores['Norm Precision'] = norm_prec_score

    tracker_disp_names = [get_tracker_display_name(trk) for trk in tracker_names]
    report_text = generate_formatted_report(tracker_disp_names, scores, table_name=report_name)
    print(report_text)


def plot_got_success(trackers, report_name):
    settings = env_settings()
    plot_draw_styles = get_plot_draw_styles()

    result_plot_path = os.path.join(settings.result_plot_path, report_name)

    auc_curve = torch.zeros((len(trackers), 101))
    scores = torch.zeros(len(trackers))

    tracker_names = []
    for trk_id, trk in enumerate(trackers):
        json_path = '{}/{}.json'.format(settings.got_reports_path, trk.name)

        if os.path.isfile(json_path):
            with open(json_path, 'r') as f:
                eval_data = json.load(f)
        else:
            raise Exception('Report not found {}'.format(json_path))

        if len(eval_data.keys()) > 1: raise Exception

        eval_data = eval_data[list(eval_data.keys())[0]]
        if 'succ_curve' in eval_data.keys():
            curve = eval_data['succ_curve']
            ao = eval_data['ao']
        elif 'overall' in eval_data.keys() and 'succ_curve' in eval_data['overall'].keys():
            curve = eval_data['overall']['succ_curve']
            ao = eval_data['overall']['ao']
        else:
            raise Exception('Invalid JSON file {}'.format(json_path))

        auc_curve[trk_id, :] = torch.tensor(curve) * 100.0
        scores[trk_id] = ao * 100.0

        tracker_names.append({'name': trk.name, 'param': trk.parameter_name, 'run_id': trk.run_id, 'disp_name': trk.display_name})

    threshold_set_overlap = torch.arange(0.0, 1.01, 0.01, dtype=torch.float64)

    success_plot_opts = {'plot_type': 'success', 'legend_loc': 'lower left', 'xlabel': 'Overlap threshold',
                         'ylabel': 'Overlap Precision [%]', 'xlim': (0, 1.0), 'ylim': (0, 100), 'title': 'Success plot'}
    plot_draw_save(auc_curve, threshold_set_overlap, scores, tracker_names, plot_draw_styles, result_plot_path,
                   success_plot_opts)
    plt.show()


def print_per_sequence_results(trackers, dataset, report_name, merge_results=False,
                               filter_criteria=None, **kwargs):
    eval_data = check_and_load_precomputed_results(trackers, dataset, report_name, **kwargs)

    if merge_results:
        eval_data = merge_multiple_runs(eval_data)

    tracker_names = eval_data['trackers']
    valid_sequence = torch.tensor(eval_data['valid_sequence'], dtype=torch.bool)
    sequence_names = eval_data['sequences']
    avg_overlap_all = torch.tensor(eval_data['avg_overlap_all']) * 100.0

    if filter_criteria is not None:
        if filter_criteria['mode'] == 'ao_min':
            min_ao = avg_overlap_all.min(dim=1)[0]
            valid_sequence = valid_sequence & (min_ao < filter_criteria['threshold'])
        elif filter_criteria['mode'] == 'ao_max':
            max_ao = avg_overlap_all.max(dim=1)[0]
            valid_sequence = valid_sequence & (max_ao < filter_criteria['threshold'])
        elif filter_criteria['mode'] == 'delta_ao':
            min_ao = avg_overlap_all.min(dim=1)[0]
            max_ao = avg_overlap_all.max(dim=1)[0]
            valid_sequence = valid_sequence & ((max_ao - min_ao) > filter_criteria['threshold'])
        else:
            raise Exception

    avg_overlap_all = avg_overlap_all[valid_sequence, :]
    sequence_names = [s + ' (ID={})'.format(i) for i, (s, v) in enumerate(zip(sequence_names, valid_sequence.tolist())) if v]

    tracker_disp_names = [get_tracker_display_name(trk) for trk in tracker_names]
    scores_per_tracker = {k: avg_overlap_all[:, i] for i, k in enumerate(tracker_disp_names)}
    report_text = generate_formatted_report(sequence_names, scores_per_tracker)
    print(report_text)


def print_results_per_video(trackers, dataset, report_name, merge_results=False,
                  plot_types=('success'), per_video=False, **kwargs):
    eval_data = check_and_load_precomputed_results(trackers, dataset, report_name, **kwargs)

    if merge_results:
        eval_data = merge_multiple_runs(eval_data)

    seq_lens = len(eval_data['sequences'])
    eval_datas = [{} for _ in range(seq_lens)]
    if per_video:
        for key, value in eval_data.items():
            if len(value) == seq_lens:
                for i in range(seq_lens):
                    eval_datas[i][key] = [value[i]]
            else:
                for i in range(seq_lens):
                    eval_datas[i][key] = value

    tracker_names = eval_data['trackers']
    valid_sequence = torch.tensor(eval_data['valid_sequence'], dtype=torch.bool)

    print('\nReporting results over {} / {} sequences'.format(valid_sequence.long().sum().item(), valid_sequence.shape[0]))

    scores = {}
    if 'success' in plot_types:
        threshold_set_overlap = torch.tensor(eval_data['threshold_set_overlap'])
        ave_success_rate_plot_overlap = torch.tensor(eval_data['ave_success_rate_plot_overlap'])
        auc_curve, auc = get_auc_curve(ave_success_rate_plot_overlap, valid_sequence)
        scores['AUC'] = auc
        scores['OP50'] = auc_curve[:, threshold_set_overlap == 0.50]
        scores['OP75'] = auc_curve[:, threshold_set_overlap == 0.75]

    if 'prec' in plot_types:
        ave_success_rate_plot_center = torch.tensor(eval_data['ave_success_rate_plot_center'])
        prec_curve, prec_score = get_prec_curve(ave_success_rate_plot_center, valid_sequence)
        scores['Precision'] = prec_score

    if 'norm_prec' in plot_types:
        ave_success_rate_plot_center_norm = torch.tensor(eval_data['ave_success_rate_plot_center_norm'])
        norm_prec_curve, norm_prec_score = get_prec_curve(ave_success_rate_plot_center_norm, valid_sequence)
        scores['Norm Precision'] = norm_prec_score

    tracker_disp_names = [get_tracker_display_name(trk) for trk in tracker_names]
    report_text = generate_formatted_report(tracker_disp_names, scores, table_name=report_name)
    print(report_text)

    if per_video:
        for i in range(seq_lens):
            eval_data = eval_datas[i]
            print('\n{} sequences'.format(eval_data['sequences'][0]))
            scores = {}
            valid_sequence = torch.tensor(eval_data['valid_sequence'], dtype=torch.bool)

            if 'success' in plot_types:
                threshold_set_overlap = torch.tensor(eval_data['threshold_set_overlap'])
                ave_success_rate_plot_overlap = torch.tensor(eval_data['ave_success_rate_plot_overlap'])
                auc_curve, auc = get_auc_curve(ave_success_rate_plot_overlap, valid_sequence)
                scores['AUC'] = auc
                scores['OP50'] = auc_curve[:, threshold_set_overlap == 0.50]
                scores['OP75'] = auc_curve[:, threshold_set_overlap == 0.75]

            if 'prec' in plot_types:
                ave_success_rate_plot_center = torch.tensor(eval_data['ave_success_rate_plot_center'])
                prec_curve, prec_score = get_prec_curve(ave_success_rate_plot_center, valid_sequence)
                scores['Precision'] = prec_score

            if 'norm_prec' in plot_types:
                ave_success_rate_plot_center_norm = torch.tensor(eval_data['ave_success_rate_plot_center_norm'])
                norm_prec_curve, norm_prec_score = get_prec_curve(ave_success_rate_plot_center_norm, valid_sequence)
                scores['Norm Precision'] = norm_prec_score

            tracker_disp_names = [get_tracker_display_name(trk) for trk in tracker_names]
            report_text = generate_formatted_report(tracker_disp_names, scores, table_name=report_name)
            print(report_text)
