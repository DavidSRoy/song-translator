from transformers import LogitsProcessor
import torch


class MyCustomLogitsProcessor(LogitsProcessor):
    def __init__(self):
        pass

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        print(input_ids.shape)
        ret_tensor = torch.full(scores.shape, 1 / (len(scores)))
        return ret_tensor
