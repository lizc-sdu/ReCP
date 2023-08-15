import argparse
import itertools
import os
import warnings

import torch
from torch.optim import lr_scheduler

from model import ReCP
from recp_data import *
from configure import get_default_config

parser = argparse.ArgumentParser()
parser.add_argument('--devices', type=str, default='0', help='gpu device ids')
parser.add_argument('--print_num', type=int, default='200', help='gap of print evaluations')
parser.add_argument('--test_time', type=int, default='10', help='number of test times')

args = parser.parse_args()


def main():
    # Environments
    warnings.filterwarnings("ignore")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.devices)
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')

    config = get_default_config()
    config['print_num'] = args.print_num

    seed = config['training']['seed']
    np.random.seed(seed)
    random.seed(seed + 1)
    torch.manual_seed(seed + 2)
    torch.cuda.manual_seed(seed + 3)
    torch.backends.cudnn.deterministic = True

    # Build the model
    recp = ReCP(config)
    optimizer = torch.optim.Adam(
        itertools.chain(recp.autoencoder_a.parameters(), recp.autoencoder_s.parameters(),
                        recp.autoencoder_d.parameters(),
                        recp.a2mo.parameters(), recp.mo2a.parameters(),
                        ), lr=config['training']['lr'], weight_decay=1e-6)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.3, verbose=False)

    recp.to_device(device)

    # Load data
    redata = ReData()
    xs_raw = [redata.a_m, redata.s_m, redata.d_m]
    xs = []
    for view in range(len(xs_raw)):
        xs.append(torch.from_numpy(xs_raw[view]).float().to(device))

    # Training
    recp.train(config, redata, xs, optimizer, scheduler, device)


if __name__ == '__main__':
    main()
