import argparse
import logging

def str2bool(s):
    return s.lower() in ("true")

def get_args():
    parser = argparse.ArgumentParser()
    opt_args = parser.add_argument_group("Optimizer")
    opt_args.add_argument("-l","--lr",type=float,default=0.001,help="learning rate")
    opt_args.add_argument("-o","--optimizer",type=str,default="adam",help="optimizer: {adam, sgd}")
    opt_args.add_argument("-b1","--beta1",type=float, default=0.9, help="beta1 for Adam")
    opt_args.add_argument("-b2","--beta2",type=float, default=0.999, help="beta2 for Adam")
    opt_args.add_argument("-m","--momentum",type=float,default=0.9,help="momentum for SGD")
    opt_args.add_argument("-w","--weight_decay",type=float,default=5e-4)
    misc_args = parser.add_argument_group("Misc")
    misc_args.add_argument("--model_name",type=str,default="AlexNetOriginal")
    misc_args.add_argument("-t","--training",type=str2bool,default=True,help="True if training, false if inference")
    misc_args.add_argument("-p","--pretrained_path",type=str,default=None,help="Path to pretrained model. Default None")
    misc_args.add_argument("-b","--batch_size",type=int,default=128, help="batch size")
    misc_args.add_argument("--save_root",type=str,default="./ckpt", help="path to save or load model(weight)")
    misc_args.add_argument("-d","--data",type=str,default="cifar10", help="path to the data, or 'cifar10' for default")
    misc_args.add_argument("--log_path",type=str, default="./logs", help="path to save log files, default './logs'")
    misc_args.add_argument("-n","--n_epoch",type=int,default=10, help="max number of epoch")
    misc_args.add_argument("--num_class",type=int,default=21)
    misc_args.add_argument("-c","--check_step",type=int,default=100)
    misc_args.add_argument("--img_size",type=int,default=224)
    misc_args.add_argument("--mode",type=str,default="finetuning",help="{from_scratch,finetuning}")
    config, _ = parser.parse_known_args()
    return config

def get_logger():
    pass
