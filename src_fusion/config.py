import time
import argparse
from sys import platform


parser = argparse.ArgumentParser(description='IVIF_Fusion')

parser.add_argument('--num_workers',
                    type=int,
                    default=16,
                    help='number of threads')
parser.add_argument('--epochs',
                    type=int,
                    default=100,
                    help='epochs for training')
parser.add_argument('--train_batch',
                    type=int,
                    default=12,
                    help='batch size for training')
parser.add_argument('--lr',
                    type=float,
                    default=0.00075,
                    help='learning rate for training')
parser.add_argument('--lr_decay_gamma',
                    type=float,
                    default=0.5,
                    help='learning rate dacay gamma value')
parser.add_argument('--lr_mstone',
                    type=list,
                    default=[50,60,70,80,90,100],
                    help='learning rate decay epoch')
parser.add_argument('--betas',
                    type=tuple,
                    default=(0.9, 0.999),
                    help='ADAM beta')
parser.add_argument('--if_warm_up',
                    type=bool,
                    default=False,
                    help='learning rate warm up during the 1st epoch')   
parser.add_argument('--save',
                    type=str,
                    default='trial',
                    help='file name to save')


args = parser.parse_args()
current_time = time.strftime('%y%m%d_%H%M%S_')
save_dir = './experiments/' + current_time + args.save
args.save_dir = save_dir

