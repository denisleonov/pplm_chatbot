import torch
import argparse
from pathlib import Path
from argparse import RawTextHelpFormatter
from utils.utils import init_seed, setup_single_gpu
from utils.decode_strategies import TopPSampling
from pplm import PPLMWithTopP

from class_data import get_dataloaders
from dataset import get_dataloaders as get_new_chat_dataloaders

from model.model import AVAILABLE_MODELS
from model.tools import build_model, get_tools, build_full_pipeline
from train import train, train_discriminator
from eval import evaluale, eval_discriminator
from interact import ChatBot
#from grid_search import run_search
import config as cfg
import wandb

def get_args():
    parser = argparse.ArgumentParser(description='trainval', formatter_class=RawTextHelpFormatter)
    parser.add_argument("-model", dest="model", choices=AVAILABLE_MODELS, default="gpt2", type=str)
    '''
    parser.add_argument("-lr", dest="lr", default=0.001, type=float,
                        help="initial learning rate")
    parser.add_argument("-epochs", dest="epochs", default=1, type=int,
                        help="number of training epochs")
    parser.add_argument("-warmup", dest="warmup", default=50, type=int,
                        help="number of warmup batches")
    parser.add_argument("-weight_decay", dest="weight_decay", default=0.005, type=float,
                        help="decay weight for regularize")
    parser.add_argument("-label_smoothing", dest="label_smoothing", default=0.1, type=float,
                        help="label smoothing")
    parser.add_argument("-ul_alpha", dest="ul_alpha", default=0.25, type=float,
                        help="mixing value for unlikelihood loss")
    parser.add_argument("-clip_value", dest="clip_value", default=1.0, type=float,
                        help="clip value")
    parser.add_argument("-batch_size", dest="batch_size", default=3, type=int,
                        help="size of training batch")
    parser.add_argument("-accumulation_interval", dest="accumulation_interval", default=1, type=int,
                        help="accumulation interval")
    parser.add_argument("-valid_interval", dest="valid_interval", default=1, type=int,
                        help="validation interval")
    '''
    parser.add_argument("-run_mode", dest="run_mode", choices=['trainval', 'test', 'interact', 'hyperopt'], default="trainval",
                        type=str, help="model run mode")
    parser.add_argument("-interact_mode", dest="interact_mode", choices=['realtime', 'test'],
                        default="realtime", type=str)
    parser.add_argument("-path2dialog", dest="path2dialog", default='', type=str)
    parser.add_argument("-mode", dest="mode", choices=['discriminator', 'generator'], default="generator",
                        type=str, help="Model mode (do you want to classify data or generate sentences?")
    parser.add_argument("-save_filename", dest="save_filename", default=None, type=Path,
                        help="file to save net weights with .pth extension")
    parser.add_argument("-load_filename", dest="load_filename", default=None,
                        type=Path, help="loading weights from file")
    parser.add_argument("-top_p", default=0.9, type=float)
    parser.add_argument("-top_k", default=0, type=float)
    parser.add_argument("-max_responce_len", default=20, type=int)
    parser.add_argument("-use_seed", default=False, type=bool)
    parser.add_argument("-history_size", default=1024, type=int)
    parser.add_argument("-temperature", default=0.7, type=float)
    #### PPLM atributes ####
    # TODO: change default values to optimal
    parser.add_argument("-colored", default=False, type=bool)
    parser.add_argument("-labels", nargs="+", default=[], type=str,
                        help="one label per head, \
                              see list of labels for head in head config.classes")
    parser.add_argument("-heads", nargs="+", default=[], type=str)
    parser.add_argument("-topics", nargs="+", default=[], type=str)
    parser.add_argument("-pplm", default=False, type=bool)
    parser.add_argument("-window_length", default=0, type=int)
    parser.add_argument("-pplm_iterations", default=3, type=int)
    parser.add_argument("-pplm_horizon", default=1, type=int)
    parser.add_argument("-pplm_stepsize", default=0.1, type=float)
    parser.add_argument("-kl_scale", default=0.01, type=float)
    parser.add_argument("-gamma", default=1.5, type=float)
    parser.add_argument("-gm_scale", default=0.9, type=float)
    parser.add_argument("-fusion_scale", default=0.9, type=float)

    return parser.parse_args()

def run_discriminator(args):
    ''' Handles all train/test stuff for one or more classification heads of model. No interact mode here.
    :param args: args from command line

    Does not support in-time heads activation/deactivation, only heads predefined in config.py now are available
    '''
    print('===> Preparing')
    if args.use_seed:
        init_seed(cfg.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device:', device)
    gpt, model, tokenizer, special_ids = build_full_pipeline('gpt2')
    model, device = setup_single_gpu(model)
    gpt, device = setup_single_gpu(gpt)
    best_score = {head: 0 for head in model.discriminator.keys()}
    print(best_score)
    if args.run_mode == 'trainval':

        optimizer, scheduler, _ = get_tools(
            model, args.mode, args.lr, args.warmup, args.weight_decay, args.clip_value,
            args.label_smoothing, 0, args.accumulation_interval, special_ids.pad
        )
        if args.load_filename is not None:
            print(f"Load model from {args.load_filename}")
            gpt.load(args.load_filename)
            model.load()
        train_loaders = {}
        valid_loaders = {}
        _ = {}
        for head in model.active_heads:
            train_loaders[head], valid_loaders[head], _[head] = get_dataloaders(tokenizer, special_ids, args.batch_size,
                                           model.discriminator[head].config['files'], head)

        for epoch in range(args.epochs):
            model, optimizer, scheduler = \
                train_discriminator(gpt, model, train_loaders, optimizer, scheduler, device, epoch, best_score=best_score)

            if not epoch % args.valid_interval:
                model, best_score = eval_discriminator(gpt, model, valid_loaders, device, best_score=best_score, epoch=epoch)
                print(f'Best scores: {best_score} (epoch: {epoch})')

    elif args.run_mode == 'test':
        test_loaders = {}
        best_score = {head: 0 for head in model.active_heads if model.discriminator[head].use == 1}

        for head in model.active_heads:
            _, _, test_loaders[head] = get_dataloaders(tokenizer, special_ids, args.batch_size,
                                           model.discriminator[head].config['files'],head)

        eval_discriminator(gpt, model, test_loaders, device, best_score)

    else:
        raise Exception(f"Wrong value for argument -mode: {args.mode}")

def run(args):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, tokenizer, special_ids = build_model(args.model)
    print('Cuda: ', torch.cuda.is_available())
    print('Model: ', model.model_name)
    print('Device: ', device)
    model = model.to(device)

    '''
    if args.run_mode == 'trainval':
    
        optimizer, scheduler, criterions = get_tools(
            model, args.mode, args.lr, args.warmup, args.weight_decay, args.clip_value,
            args.label_smoothing, args.ul_alpha, args.accumulation_interval, special_ids.pad)

        model, device = setup_single_gpu(model)
        
        if args.load_filename is not None:
            print(args.load_filename)
            model.load(args.load_filename)
        
        train_loader, valid_loader = get_new_chat_dataloaders(tokenizer, args.model, args.batch_size)

        generator = TopPSampling(model, model.special_ids, device, args.max_responce_len, top_p=args.top_p)

        rouge = float('-inf')

        for epoch in range(args.epochs):

            model, optimizer, scheduler = \
                train(model, train_loader, optimizer, scheduler, criterions, device, epoch=epoch)

            if not epoch % args.valid_interval:
                if epoch != 0:
                    model, rouge = evaluale(model, valid_loader, device, generator, args.save_filename, rouge, max_iter=100)

    elif args.run_mode == 'test':

        model, device = setup_single_gpu(model)
        generator = TopPSampling(model, model.special_ids, device, args.max_responce_len, top_p=args.top_p)
        _, _, test_loader = get_dataloaders(tokenizer, special_ids, args.batch_size)
        evaluale(model, test_loader, device, generator, args.save_filename)
    '''
    if args.run_mode == 'interact':
        model, device = setup_single_gpu(model)
        model.eval()
        model.use_cache(True)

        if args.load_filename is not None:
            print(f'using weights from {args.load_filename}')
            model.load(args.load_filename)
        if not args.pplm:
            print('- Interact without PPLM')
            generator = TopPSampling(model, model.special_ids, device, args.max_responce_len, top_p=args.top_p)
        else:
            # TODO: change and test generator
            print(args.labels)
            print('+ Interact with PPLM')
            generator = PPLMWithTopP(model, model.special_ids, device, args.max_responce_len,
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
                                          fusion_scale=args.fusion_scale)
        bot = ChatBot(model, device, generator, args.history_size)
        if args.interact_mode == 'test':
            bot.run_test(path2dialog=args.path2dialog, 
                         colored=args.colored)
        else:
            bot.init_context()
            bot.interact(pplm=args.pplm, 
                         colored=args.colored)

    #elif args.run_mode == 'hyperopt':
    #    run_search(args)
    else:
        raise Exception(f"Wrong value for argument -mode: {args.mode}")

def main():

    args = get_args()

    if args.use_seed:
        init_seed(cfg.SEED)

    if args.mode == 'discriminator':
        print(f'I use {cfg.HEADS} data and classify it')
        run_discriminator(args)
    if args.mode == 'generator':
        print('I use persona data and generate sentences')
        run(args)

if __name__ == '__main__':
    main()
