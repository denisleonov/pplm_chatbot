import os
import torch

from transformers.optimization import AdamW

from src.model.attribute_model import AttributeModel
from src.model.model import DoubleHeadsModel
from src.utils.criterions import SmoothLoss, UnlikelihoodLoss
from src.utils.scheduler import NoamScheduler

REQUIRED_LOCAL_FILES = [
    'pytorch_model.bin',
    'config.json',
    'vocab.json',
    'merges.txt'
]

def build_model(model_name, use_cls_head=False, use_cache=False):
    """Builds pretrained transformer-based model with LM head, tokeniser and namedtuple with special ids.
   
    :return: model, tokenizer, special_ids
    """
    
    # load model from local files if possible
    if model_name.endswith('-local'):
        if os.path.isdir(model_name):
            for filename in REQUIRED_LOCAL_FILES:
                fullname = os.path.join(model_name, filename)
                if not os.path.isfile(fullname):
                    print(f"Can't load the model with name {model_name}, {filename} is required. Trying to load the model from net.")
                    model_name = model_name.replace('-local', '')
                    break
            else:
                print(f"Loading the model from local dir {model_name}.")
        else:
            print(f"Can't load the model with name {model_name}, there is no local folder for the model.")
            model_name = model_name.replace('-local', '')
            print(f"Trying to load the model {model_name} from net.")

    else:
        print(f"Trying to load the model {model_name} from net.")

    tokenizer, special_ids = DoubleHeadsModel.build_tokenizer(model_name)
    model = DoubleHeadsModel(model_name, tokenizer, special_ids, use_cls_head, use_cache=use_cache)

    return model, tokenizer, special_ids

def build_attrmodel_from_model(model, args, device=torch.device('cpu')):
    attr_model = AttributeModel(model.hidden_size, model.vocab_size, args, model.tokenizer, device)
    return attr_model

def build_full_pipeline(args,
                        model_name='gpt2',
                        heads=('sentiment', ),
                        device=torch.device('cpu'),
                        full_init=True):
    # TODO: rewrite for new architecture
    if type(heads) != tuple:
        heads = (heads, )
    model, tokenizer, special_ids = build_model(model_name)
    attr_model = build_attrmodel_from_model(model, args, device)
    attr_model.initialize_discriminator(heads, full_init)
    return model, attr_model, tokenizer, special_ids

def get_tools(model, mode, lr, warmup, weight_decay,
              label_smoothing, ul_alpha, pad_id):

    optimizer = AdamW([
            {'params': [param for param in model.parameters() if param.requires_grad], 'lr': lr},
        ], weight_decay=weight_decay)

    scheduler = NoamScheduler(optimizer, warmup=warmup)

    criterions = [SmoothLoss(label_smoothing, model.vocab_size, pad_id)]
    if ul_alpha > 0:
        criterions.append(UnlikelihoodLoss(pad_id, ul_alpha))
    return optimizer, scheduler, criterions
