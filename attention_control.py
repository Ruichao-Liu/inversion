from typing import Optional, Union, Tuple, List, Callable, Dict
from tqdm.notebook import tqdm
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch.nn.functional as nnf
import numpy as np
import abc
import ptp_utils
import seq_aligner
import shutil
from torch.optim.adam import Adam
from PIL import Image
LOW_RESOURCE = False

class AttentionControl(abc.ABC):

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
            # self.step_store[self.cur_step][key] = []
            # self.step_store[self.cur_step][key].append(attn.detach().cpu())
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in
                             self.attention_store}
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}




class AttentionExtractor:
    def __init__(self):
        self.attention_maps = {}  # 保存注意力图的字典

    def hook_attention(self, module, input, output):

        # 注意力权重通常在输出张量中，例如 output 是 (batch, heads, tokens, tokens)
        self.attention_maps[self.current_step] = output.detach().cpu()  # 保存到字典中

    def extract_attention(self, model, latents, timesteps, tokenizer=None):
        """
        主函数，用于执行扩散模型的每一步推理，并提取注意力图。
        """
        # 遍历模型中的注意力层，注册 forward hook
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.MultiheadAttention):  # 替换为你的注意力模块类
                hooks.append(module.register_forward_hook(self.hook_attention))

        self.attention_maps = {}  # 重置注意力图存储
        for step, t in enumerate(timesteps):
            self.current_step = t
            # 在当前时间步进行推理
            with torch.no_grad():
                _ = model(latents, t)  # 替换为你的模型调用方式

        # 清理 hook
        for hook in hooks:
            hook.remove()

        return self.attention_maps
