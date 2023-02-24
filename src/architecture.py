from fedscale.core.model_manager import SuperModel
from argparse import Namespace
from copy import deepcopy


def get_model(model, shrink_rate):
    ns = Namespace(**{"task": "vision", "data_set": "cifar10"})
    super_model = SuperModel(model, ns, 0)
    new_model = super_model.model_width_scale(ratio=shrink_rate)
    return new_model


class Architecture:
    def __init__(self, torch_model, shrink_rates: list = []):
        self.models = {0: deepcopy(torch_model)}
        for rate in shrink_rates:
            self.models[rate] = get_model(torch_model, rate)

    def get_model(self, rate):
        return self.models[rate]