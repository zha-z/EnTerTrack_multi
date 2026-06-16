from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/data2/got10k_lmdb'
    settings.got10k_path = '/data2/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/data/itb'
    settings.lasot_extension_subset_path = '/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/data2/lasot_lmdb'
    settings.lasot_path = '/data2/lasot'
    settings.network_path = '/data/zjy/multi/output'    # Where tracking networks are stored.
    settings.nfs_path = '/data/nfs'
    settings.otb_path = '/data/otb'
    settings.prj_dir = '/data/zjy/EnTeR-Track-main'
    settings.result_plot_path = '/data/zjy/EnTeR-Track-main/output/test/result_plots'
    settings.results_path = '/data/zjy/EnTeR-Track-main/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/data/zjy/multi/output/entertrack_single_lasot_ft_cons'
    settings.segmentation_path = '/data/zjy/EnTeR-Track-main/output/test/segmentation_results'
    settings.tc128_path = '/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/data/tnl2k/TNL2K_test_subset'
    settings.tpl_path = ''
    settings.trackingnet_path = '/data2/trackingnet'
    settings.uav_path = '/data/uav123'
    settings.vot18_path = '/data/vot2018'
    settings.vot22_path = '/data/vot2022'
    settings.vot_path = '/data/VOT2019'
    settings.youtubevos_dir = ''
    settings.dtb_path = '/data/DTB70'


    settings.mdot_test_path = '/data2/Two-MDOT/test/two/'
    settings.threemdot_test_path = '/data2/Three-MDOT/three'

    settings.dtb70_path = '/data/DTB70'
    settings.uavdt_path = '/data/UAVDT'
    settings.visdrone_path = '/data/VisDrone2018'
    settings.uav123_10fps_path = '/data/UAV123_10fps'
    settings.uav123_path = '/data/uav123'
    settings.uavtrack_path = '/data/UAVTrack112'
    return settings

