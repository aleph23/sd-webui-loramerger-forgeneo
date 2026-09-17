
import os
import os.path
import torch
import gradio as gr
from modules import script_callbacks, sd_models,sd_models
from modules.scripts import basedir
import scripts.mergers.pluslora as pluslora

if hasattr(sd_models, "checkpoints_loaded"):
    checkpoints_loaded = sd_models.checkpoints_loaded
    load_model = sd_models.load_model
else:
    checkpoints_loaded = sd_models.checkpoints_list
    load_model = None

CALCMODES  = ["normal", "cosineA", "cosineB","trainDifference","smoothAdd","smoothAdd MT","extract","tensor","tensor2","self","plus random"]

def sorted_checkpoint_tiles():
    return sorted(sd_models.checkpoint_tiles(), key=lambda name: name.casefold())

from typing import Union
def network_reset_cached_weight(self: Union[torch.nn.Conv2d, torch.nn.Linear]):
    self.network_current_names = ()
    self.network_weights_backup = None
    self.network_bias_backup = None
    

def fix_network_reset_cached_weight():
    try:
        import networks as net
        net.network_reset_cached_weight = network_reset_cached_weight
    except:
        pass

def on_ui_tabs():
    fix_network_reset_cached_weight()


    with gr.Blocks() as supermergerui:
        # LoRAタブを削除し、直接コンテンツを表示
        pluslora.on_ui_tabs()
        import lora          

    return (supermergerui, "LoraMerge", "loramerge"),

if __package__ == "supermerger":
    script_callbacks.on_ui_tabs(on_ui_tabs)
