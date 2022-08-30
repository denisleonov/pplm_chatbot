import argparse

import omegaconf
import torch

from src.utils.decode_strategies import TopPSampling
from src.interact import ChatBot
from src.pplm import PPLMWithTopP
from src.model.tools import build_model
from src.utils.utils import read_heads_only_configs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-pplm_config', default='configs/pplm.yaml', type=str)
    parser.add_argument('-device', default='cpu', choices=['cpu', 'cuda'], type=str)
    parser.add_argument('-chatbot_checkpoint_path', default='./model_weights/bart_exp1_ep49.ckpt', type=str)
    parser_args = parser.parse_args()
    args = omegaconf.OmegaConf.load(parser_args.pplm_config)
    head_cfgs = read_heads_only_configs()
    args = omegaconf.OmegaConf.merge(args, head_cfgs)

    model, tokenizer, special_ids = build_model(args.model, use_cache=args.use_cache, use_cls_head=args.use_cls_head)

    if parser_args.chatbot_checkpoint_path != '':
        print(f'Load model from checkpoint {parser_args.chatbot_checkpoint_path}')
        ckpt = torch.load(parser_args.chatbot_checkpoint_path, map_location=torch.device('cpu'))
        state_dict = {k.replace('model.', ''): v for k, v in ckpt['state_dict'].items()}
        model.load_state_dict(state_dict)
    device = torch.device(parser_args.device)
    model = model.to(device)
    model.eval()

    if not args.pplm:
        print('- Interact without PPLM')
        generator = TopPSampling(model, special_ids, device, args.max_responce_len, top_p=args.top_p)
    else:
        print('+ Interact with PPLM')
        # TODO: what do heads and labels should do?
        args['heads'] = ('dd_topics', 'sentiment')
        args['labels'] = ('Work', 'anger')
        generator = PPLMWithTopP(
            args,
            model,
            special_ids,
            device,
            args.max_responce_len,
            top_p=args.top_p,
            labels=args.labels,
            heads=args.heads,
            topics=args.topics,
            temperature=args.temperature,
            window_length=args.window_length,
            iterations=args.pplm_iterations,
            horizon_length=args.pplm_horizon,
            stepsize=args.pplm_stepsize,
            kl_scale=args.kl_scale,
            gamma=args.gamma,
            fusion_scale=args.fusion_scale
        )
    bot = ChatBot(model, device, generator, args.history_size)
    bot.interact(
        pplm=args.pplm,
        colored=args.colored
    )
