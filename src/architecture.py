from fedscale.core.model_manager import SuperModel
from argparse import Namespace
from copy import deepcopy
from thop import profile
import torch


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
        print("complete preparing architectures")
        input = torch.randn(10, 3, 32, 32)
        for rate in shrink_rates:
            macs, params = profile(self.models[rate], inputs=(input,), verbose=False)
            print(f"shrink rate: {rate}, macs: {macs}")

    def get_model(self, rate):
        return self.models[rate]