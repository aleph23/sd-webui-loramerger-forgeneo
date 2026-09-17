import gc
import torch
import tqdm
from typing import Optional
import json
import launch
from tqdm import tqdm
from torch import Tensor, lerp
from torch.nn.functional import cosine_similarity
from modules import sd_models
try:
    from modules.infotext_utils import create_override_settings_dict
except ImportError:
    from modules.generation_parameters_copypaste import create_override_settings_dict

PREFIXFIX = ("double_blocks","single_blocks","time_in","vector_in","txt_in")
PREFIX_M = "model.diffusion_model."
BNB = ".quant_state.bitsandbytes__"

try:
    ui_version = int(launch.git_tag().split("-",1)[0].replace("v","").replace(".",""))
except:
    ui_version = 100

try:
    from modules import launch_utils
    forge = launch_utils.git_tag()[0:2] == "f2"
    reforge = launch_utils.git_tag()[0:2] == "f1" or launch_utils.git_tag() == "classic" 
    neo = launch_utils.git_tag() == "neo"
except:
    forge = reforge = neo = False

if forge or neo:
    from backend import memory_management
    from backend.utils import load_torch_file


from inspect import currentframe







#type[0:aplha,1:beta,2:seed,3:mbw,4:model_A,5:model_B,6:model_C]
#msettings=[0 weights_a,1 weights_b,2 model_a,3 model_b,4 model_c,5 base_alpha,6 base_beta,7 mode,8 useblocks,9 custom_name,10 save_sets,11 id_sets,12 wpresets]
#id sets "image", "PNG info","XY grid"



#msettings=[weights_a,weights_b,model_a,model_b,model_c,device,base_alpha,base_beta,mode,loranames,useblocks,custom_name,save_sets,id_sets,wpresets,deep]  








# XXX hack. fake checkpoint_info

BLOCKIDFLUX = ["CLIP", "T5", "IN"] + ["D{:002}".format(x) for x in range(19)] + ["S{:002}".format(x) for x in range(38)] + ["OUT"] # Len: 61


    
################################################
##### Main Merging Code


################################################
##### cosineA/B


################################################
##### Traindiff

################################################
##### Extract
def extract_super(base: Optional[Tensor], a: Tensor, b: Tensor, alpha: float, beta: float, gamma: float) -> Tensor:
    assert base is None or base.shape == a.shape
    assert a.shape == b.shape
    assert 0 <= alpha <= 1
    assert 0 <= beta <= 1
    assert 0 <= gamma
    dtype = base.dtype if base is not None else a.dtype
    base = base.float() if base is not None else 0
    a = a.float() - base
    b = b.float() - base
    c = cosine_similarity(a, b, -1).clamp(-1, 1).unsqueeze(-1)
    d = ((c + 1) / 2) ** gamma
    result = base + lerp(a, b, alpha) * lerp(d, 1 - d, beta)
    return result.to(dtype)


################################################
##### Tensor Merge

################################################
##### Multi Thread SmoothAdd


################################################
##### Elementals


################################################
##### Load Model





################################################
##### Logging













################################################
##### Random





################################################
##### Generate Image



################################################
##### Block Ids




################################################
##### Assert Inpaint

################################################
##### Adjust






################################################
##### Include/Exclude


################################################
##### Reset Broken CliP IDs



################################################
##### cache



################################################
##### print




################################################
##### model_loader

################################################
##### forge
def unload_forge():
    sd_models.model_data.sd_model = None
    sd_models.model_data.loaded_sd_models = []
    memory_management.unload_all_models()
    memory_management.soft_empty_cache()
    gc.collect()

def prefixer(t, revert = False):
    keys = list(t.keys())
    if revert: 
        for key in keys:
            t[key.replace(PREFIX_M,"")] = t.pop(key)
        print('"model.diffusion_model." removed from prifix.')
        return

    need_revert = False
    for key in keys:
        if key.startswith(PREFIXFIX):
            t["model.diffusion_model." + key] = t.pop(key)
            need_revert = True
    if need_revert:
        print('"model.diffusion_model." added to prifix.')
    gc.collect()
    return need_revert


###############################################################
######## QLoRA   
def qdtyper(sd):
    if any("fp4" in k for k in sd):
        return "fp4"
    elif any("nf4" in k for k in sd):
        return "nf4"
    for key in sd:
        if hasattr(sd[key],"dtype"):
            return sd[key].dtype


    
def q_dequantize(sd,qtype,device,dtype,setbnb = True):
    dellist = [] 
    from bitsandbytes.functional import dequantize_4bit
    for key in tqdm(sd):
        if ("weight" in key) and ("weight." not in key) and (key + BNB + qtype in sd):
            qs = q_tensor_to_dict(sd[key + BNB + qtype])
            out = torch.empty(qs["shape"],device="cuda:0")
            sd[key] = dequantize_4bit(sd[key].to("cuda:0"),out=out, absmax=sd[key + ".absmax"].to("cuda:0"),blocksize=qs["blocksize"],quant_type=qs["quant_type"]).to(device,dtype)
            dellist.append(key + ".absmax")
            if setbnb:dellist.append(key + BNB + qtype)
            dellist.append(key + ".quant_map")
        elif isinstance(sd[key], torch.Tensor):
            sd[key] = sd[key].to(dtype)

    for key in dellist:
        if key in sd:
            del sd[key]

def q_quantize(sd:dict,qtype,device,setbnb = True):
    from bitsandbytes.functional import quantize_4bit
    sd_plus = {}
    for key in tqdm(sd):
        if "weight" in key and "weight." not in key:
            weight, state = quantize_4bit(sd[key].to("cuda:0"), quant_type=qtype)
            sd[key] = weight.to(device)
            sd_plus[key + ".absmax"] = state.absmax
            sd_plus[key + ".quant_map"] = state.code
            if setbnb: sd_plus[key + BNB + qtype] = state.as_dict(True)["quant_state." + "bitsandbytes__" + qtype]
    sd.update(sd_plus)

def q_tensor_to_dict(tensor):
    num_list = tensor.tolist()
    char_list = [chr(num) for num in num_list]
    json_string = ''.join(char_list)

    tensor_dict = json.loads(json_string)
    return tensor_dict
