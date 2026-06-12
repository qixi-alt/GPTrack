import os


class EnvironmentSettings:
    """Local training paths for GPTrack."""

    def __init__(self):
        self.workspace_dir = "/home/dell/data/Liyutong/RGBT_tracking/GPTrack"
        self.tensorboard_dir = os.path.join(self.workspace_dir, "tensorboard")
        self.pretrained_networks = os.path.join(self.workspace_dir, "pretrained_networks")

        data_dir = os.path.join(self.workspace_dir, "data")

        # Training dataset.
        self.lasher_train_dir = os.path.join(data_dir, "lasher", "trainingset")
        self.lasher_test_dir = os.path.join(data_dir, "lasher", "testingset")

        # Evaluation datasets kept for path consistency.
        self.rgbt210_dir = os.path.join(data_dir, "rgbt210")
        self.rgbt234_dir = os.path.join(data_dir, "rgbt234") + os.sep
        self.gtot_dir = os.path.join(data_dir, "gtot")
        self.UAV_RGBT_dir = os.path.join(data_dir, "VTUAV", "Test")
