from typing import List, Tuple, Dict
import torch
import torch.nn as nn


@torch.no_grad()
def aggregate_models_state_dict(client_models: List[nn.Module],
                                destination_module: nn.Module,
                                aggregation_weights: List[float] = None) ->None:
    if aggregation_weights is None:
        aggregation_weights = [1 / len(client_models)] * len(client_models)
    res = {k: torch.zeros_like(v) for k, v in client_models[0].state_dict().items() if "num_clusters" not in k}
    for w, model in zip(aggregation_weights, client_models):
        for k, v in model.state_dict().items():
            if "num_batches_tracked" in k:
                continue
            res[k] += w * v.data.clone()
    for k, v in destination_module.state_dict().items():
        if "num_batches_tracked" in k:
            continue
        v.data.copy_(res[k])


def aggregate_models_parameters(client_models: List[nn.Module],
                                destination_module: nn.Module,
                                aggregation_weights: List[float] = None) ->None:
    if aggregation_weights is None:
        aggregation_weights = [1 / len(client_models)] * len(client_models)
    res = {k: torch.zeros_like(v) for k, v in client_models[0].named_parameters()}
    for w, model in zip(aggregation_weights, client_models):
        for k, v in model.named_parameters():
            res[k] += w * v.data.clone()
    for k, v in destination_module.named_parameters():

        v.data.copy_(res[k])
