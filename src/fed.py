import copy
import numpy as np
import torch
from collections import OrderedDict


class Federation:
    def __init__(self, num_users, global_parameters, rate):
        self.num_users = num_users
        self.global_parameters = global_parameters
        self.rate = rate
        self.idx = self.split_model()

    def split_model(self):
        idx_i = [None for _ in range(self.num_users)]
        idx = [OrderedDict() for _ in range(self.num_users)]
        output_weight = [k for k in self.global_parameters.keys() if 'weight' in k][-1]
        for k, v in self.global_parameters.items():
            parameter_type = k.split('.')[-1]
            for m in range(self.num_users):
                if parameter_type in ['weight', 'bias']:
                    if parameter_type == 'weight':
                        if v.dim() > 1:
                            input_size = v.size(1)
                            output_size = v.size(0)
                            if idx_i[m] is None:
                                idx_i[m] = torch.arange(input_size, device=v.device)
                            input_idx_i_m = idx_i[m]
                            if k == output_weight:
                                output_idx_i_m = torch.arange(output_size, device=v.device)
                            else:
                                local_output_size = int(np.ceil(output_size * self.rate))
                                output_idx_i_m = torch.randperm(output_size, device=v.device)[:local_output_size]
                            idx[m][k] = torch.meshgrid(output_idx_i_m, input_idx_i_m)
                            idx_i[m] = output_idx_i_m
                        else:
                            input_idx_i_m = idx_i[m]
                            idx[m][k] = input_idx_i_m
                    else:
                        input_idx_i_m = idx_i[m]
                        idx[m][k] = input_idx_i_m
        return idx

    def distribute(self, user_idx):
        local_parameters = [OrderedDict() for _ in range(len(user_idx))]
        for k, v in self.global_parameters.items():
            parameter_type = k.split('.')[-1]
            for m in range(len(user_idx)):
                if parameter_type in ['weight', 'bias']:
                    if parameter_type == 'weight':
                        if v.dim() > 1:
                            local_parameters[m][k] = copy.deepcopy(v[self.idx[user_idx[m]][k]])
                        else:
                            local_parameters[m][k] = copy.deepcopy(v[self.idx[user_idx[m]][k]])
                    else:
                        local_parameters[m][k] = copy.deepcopy(v[self.idx[user_idx[m]][k]])
                else:
                    local_parameters[m][k] = copy.deepcopy(v)
        return local_parameters

    def combine(self, local_parameters, user_idx):
        count = OrderedDict()
        for k, v in self.global_parameters.items():
            parameter_type = k.split('.')[-1]
            count[k] = v.new_zeros(v.size(), dtype=torch.float32)
            tmp_v = v.new_zeros(v.size(), dtype=torch.float32)
            for m in range(len(local_parameters)):
                if parameter_type in ['weight', 'bias']:
                    tmp_v[self.idx[user_idx[m]][k]] += local_parameters[m][k]
                    count[k][self.idx[user_idx[m]][k]] += 1
                else:
                    tmp_v += local_parameters[m][k]
                    count[k] += 1
            tmp_v[count[k] > 0] = tmp_v[count[k] > 0].div_(count[k][count[k] > 0])
            v[count[k] > 0] = tmp_v[count[k] > 0].to(v.dtype)
        return