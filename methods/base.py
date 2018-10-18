import torch
import abc

class AttributionMethod:
    """ Something than can make a methods heatmap from a model plus input"""
    @abc.abstractmethod
    def get_map(self, img: torch.Tensor, target: torch.Tensor):
        pass

    @abc.abstractmethod
    def name(self):
        pass
