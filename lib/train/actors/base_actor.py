from lib.utils import TensorDict
from lib.train.pcum_freeze import (
    assert_pcum_frozen_batchnorm_eval,
    set_pcum_frozen_modules_eval,
)


class BaseActor:
    """ Base class for actor. The actor class handles the passing of the data through the network
    and calculation the loss"""
    def __init__(self, net, objective):
        """
        args:
            net - The network to train
            objective - The loss function
        """
        self.net = net
        self.objective = objective

    def __call__(self, data: TensorDict):
        """ Called in each training iteration. Should pass in input data through the network, calculate the loss, and
        return the training stats for the input data
        args:
            data - A TensorDict containing all the necessary data blocks.

        returns:
            loss    - loss for the input data
            stats   - a dict containing detailed losses
        """
        raise NotImplementedError

    def to(self, device):
        """ Move the network to device
        args:
            device - device to use. 'cpu' or 'cuda'
        """
        self.net.to(device)

    def train(self, mode=True):
        """ Set whether the network is in train mode.
        args:
            mode (True) - Bool specifying whether in training mode.
        """
        self.net.train(mode)
        cfg = getattr(self, "cfg", None)
        if mode and cfg is not None:
            set_pcum_frozen_modules_eval(self.net, cfg)
            assert_pcum_frozen_batchnorm_eval(self.net, cfg)

    def eval(self):
        """ Set network to eval mode"""
        self.train(False)
