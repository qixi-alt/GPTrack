import os

from lib.test.evaluation.environment import EnvSettings


def local_env_settings():
    """Local evaluation paths for GPTrack."""

    settings = EnvSettings()

    base_dir = "/home/dell/data/Liyutong/RGBT_tracking/GPTrack"
    data_dir = os.path.join(base_dir, "data")
    output_dir = os.path.join(base_dir, "output")

    settings.prj_dir = base_dir
    settings.save_dir = output_dir
    settings.check_dir = "checkpoints"
    settings.network_path = os.path.join(base_dir, "pretrained")
    settings.results_path = os.path.join(output_dir, "test", "tracking_results")
    settings.result_plot_path = os.path.join(output_dir, "test", "result_plots")
    settings.segmentation_path = os.path.join(output_dir, "test", "segmentation_results")

    # Evaluation datasets.
    settings.lasher_path = os.path.join(data_dir, "lasher")
    settings.rgbt210_dir = os.path.join(data_dir, "rgbt210")
    settings.rgbt234_dir = os.path.join(data_dir, "rgbt234") + os.sep
    settings.gtot_dir = os.path.join(data_dir, "gtot")
    settings.UAV_RGBT_dir = os.path.join(data_dir, "VTUAV", "Test")

    return settings
