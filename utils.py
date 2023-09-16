import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', help='experiement name', type=str, default='MAIC')
    parser.add_argument('--model', help='Model'  , type=str, default='CNN', choices=['CNN'])
    parser.add_argument('--epochs' , type=int, default=100)
    parser.add_argument('--batch_size' , type=int, default=128)
    parser.add_argument('--fold' , type=int, default=1, choices=[1,2,3,4,5])
    
    parser.add_argument('--seed', type=int, default=0)
     
    parser.add_argument('--use_tb', type=str2bool, default=False)

    args = parser.parse_args()
    return args

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
    
