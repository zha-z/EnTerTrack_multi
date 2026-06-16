import _init_paths
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]

from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist

trackers = []
dataset_name = 'threemdot_test'

trackers.extend(trackerlist(name='entertrack', parameter_name='entertrack_threemdot', dataset_name=dataset_name,
                            run_ids=1, display_name='entertrack_base_300'))
trackers.extend(trackerlist(name='entertrack', parameter_name='entertrack_threemdot', dataset_name=dataset_name,
                            run_ids=2, display_name='entertrack_base_100'))
trackers.extend(trackerlist(name='entertrack', parameter_name='entertrack_threemdot', dataset_name=dataset_name,
                            run_ids=3, display_name='entertrack_base_80'))
trackers.extend(trackerlist(name='entertrack', parameter_name='entertrack_threemdot', dataset_name=dataset_name,
                            run_ids=4, display_name='entertrack_base_60'))
trackers.extend(trackerlist(name='entertrack', parameter_name='entertrack_threemdot', dataset_name=dataset_name,
                            run_ids=5, display_name='entertrack_base_80_0.0004'))
trackers.extend(trackerlist(name='entertrack', parameter_name='entertrack_threemdot', dataset_name=dataset_name,
                            run_ids=6, display_name='entertrack_base_31_0.0004'))
trackers.extend(trackerlist(name='entertrack', parameter_name='entertrack_threemdot', dataset_name=dataset_name,
                            run_ids=7, display_name='entertrack_base_60_0.0004'))
trackers.extend(trackerlist(name='entertrack', parameter_name='entertrack_threemdot', dataset_name=dataset_name,
                            run_ids=8, display_name='entertrack_base_48_0.0004_50_10_25'))
trackers.extend(trackerlist(name='entertrack', parameter_name='entertrack_threemdot', dataset_name=dataset_name,
                            run_ids=9, display_name='entertrack_base_lasot_10_0.0004_50_10_25'))
trackers.extend(trackerlist(name='entertrack', parameter_name='entertrack_threemdot', dataset_name=dataset_name,
                            run_ids=10, display_name='entertrack_base_lasot_50_0.0004_50_10_25'))
trackers.extend(trackerlist(name='entertrack_teacher', parameter_name='entertrack_teacher', dataset_name=dataset_name,
                            run_ids=7, display_name='teacher'))

trackers.extend(trackerlist(name='entertrack', parameter_name='entertrack_threemdot', dataset_name=dataset_name,
                            run_ids=11, display_name='single'))
trackers.extend(trackerlist(name='entertrack', parameter_name='entertrack_threemdot', dataset_name=dataset_name,
                            run_ids=12, display_name='single_30'))
trackers.extend(trackerlist(name='entertrack', parameter_name='entertrack_threemdot', dataset_name=dataset_name,
                            run_ids=13, display_name='single_21'))
# trackers.extend(trackerlist(name='entertrack', parameter_name='entertrack_threemdot_lasot_ft', dataset_name=dataset_name,
#                             run_ids=14, display_name='single_lasot_ft_4'))

trackers.extend(trackerlist(name='entertrack', parameter_name='pcum_ablation_baseline', dataset_name=dataset_name,
                            run_ids=0, display_name='pcum_ablation_baseline'))
trackers.extend(trackerlist(name='entertrack', parameter_name='pcum_ablation_local_film', dataset_name=dataset_name,
                            run_ids=0, display_name='pcum_ablation_local_film'))
trackers.extend(trackerlist(name='entertrack', parameter_name='pcum_ablation_local_gated', dataset_name=dataset_name,
                            run_ids=0, display_name='pcum_ablation_local_gated'))
trackers.extend(trackerlist(name='entertrack', parameter_name='pcum_ablation_pseudo_align_film', dataset_name=dataset_name,
                            run_ids=0, display_name='pcum_ablation_pseudo_align_film'))
trackers.extend(trackerlist(name='entertrack', parameter_name='pcum_ablation_pseudo_align_gated', dataset_name=dataset_name,
                            run_ids=0, display_name='pcum_ablation_pseudo_align_gated'))

trackers.extend(trackerlist(name='entertrack', parameter_name='pcum_real_target_stable', dataset_name=dataset_name,
                            run_ids=0, display_name='pcum_real_target_stable'))
trackers.extend(trackerlist(name='entertrack', parameter_name='pcum_real_target_stable', dataset_name=dataset_name,
                            run_ids=1, display_name='pcum_real_target_stable_28'))

trackers.extend(trackerlist(name='entertrack', parameter_name='pcum_real_target_stable', dataset_name=dataset_name,
                            run_ids=2, display_name='pcum_real_target_stable_V2'))
trackers.extend(trackerlist(name='entertrack', parameter_name='pcum_real_allviews_stable', dataset_name=dataset_name,
                            run_ids=0, display_name='pcum_real_allviews_stable_28'))
trackers.extend(trackerlist(name='entertrack', parameter_name='pcum_real_allviews_a_focus', dataset_name=dataset_name,
                            run_ids=0, display_name='pcum_real_allviews_a_focus_40'))
# trackers.extend(trackerlist(name='entertrack', parameter_name='entertrack_threemdot', dataset_name=dataset_name,
#                             run_ids=1, display_name='entertrack_lasot100'))
dataset = get_dataset(dataset_name)


#print_per_sequence_results(trackers, dataset, dataset_name, merge_results=False)
print_results(trackers, dataset, dataset_name, merge_results=False, plot_types=('success', 'norm_prec', 'prec'))
# print_results(trackers, dataset, 'UNO', merge_results=True, plot_types=('success', 'prec'))
