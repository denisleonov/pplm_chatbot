import random
import os
import torch
import numpy as np
import omegaconf

def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_single_gpu(model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    model = model.to(device)
    model.device = device
    return model, device

def add_to_sequence(seq, elem):
    return torch.cat([seq, torch.tensor([[elem]]).type_as(seq)], dim=-1)

def test_device(device: str):
    torch.ones(1,2,3).to(device)

def to_numpy(target):
    "Convertes torch tensor to numpy array"
    return target.detach().cpu().numpy()

def read_chatbot_configs(path_to_model_cfg: str,
                         path_to_cfgs_dir: str = './configs/chatbot_configs/'
                        ):
    exp_path = path_to_cfgs_dir + 'experiments/'
    path = path_to_cfgs_dir + path_to_model_cfg
    files = [path_to_cfgs_dir+fl for fl in os.listdir(path_to_cfgs_dir) if 'yaml' in fl]
    base_cfg = [omegaconf.OmegaConf.load(cfg) for cfg in files]
    base_cfg = omegaconf.OmegaConf.merge(*base_cfg)

    exp_cfg = omegaconf.OmegaConf.load(exp_path + path_to_model_cfg)
    cfg = omegaconf.OmegaConf.merge(base_cfg, exp_cfg)
    return cfg

def read_discriminator_configs(
        path_to_experiment_cfg: str,
        path_to_cfgs_dir: str = './configs/discriminator_configs/'
    ):
    cfg = read_chatbot_configs(path_to_experiment_cfg, path_to_cfgs_dir)
    head_cfg = omegaconf.OmegaConf.load(path_to_cfgs_dir + 'head_configs/{}.yaml'.format(cfg.name))
    cfg = omegaconf.OmegaConf.merge(head_cfg, cfg)
    return cfg

def read_heads_only_configs(
        path_to_cfgs_dir: str = './configs/discriminator_configs/'
    ):
    files = os.listdir(path_to_cfgs_dir + 'head_configs/')
    files = [fl for fl in files if 'yaml' in fl]
    cfg = omegaconf.OmegaConf.merge(
        *[
            omegaconf.OmegaConf.load(path_to_cfgs_dir + 'head_configs/{}'.format(f)) for f in files
        ]
    )
    return cfg
    
def apply_threshold(preds, treshold=0.5):
    return (preds > treshold).float()