import os

import numpy as np
from tqdm import tqdm

import yaml
from yaml import CLoader as Loader
from munch import munchify
from training_strategy.self_supervised_learning.vicreg import *
from training_strategy.self_supervised_learning.augmentations import *

from utils_3d.dataset import PatchUnlabeledDataset
from utils import fix_random_seed


def parse_args() -> dict:
    import argparse
    """Read commandline arguments
    Argument list includes mostly tunable hyper-parameters (learning rate, number of epochs, etc).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg", "--config", default='vicreg_patch.yml',
                        help="Path for the config file")
    parser.add_argument("--exp-suffix", default='', type=str,
                        help="Suffix in the experiment name")
    parser.add_argument("--optimizer", default='lars', type=str,
                        help="optimizer")
    parser.add_argument("--base-lr", default=0.2, type=float,
                        help="learning rate")
    parser.add_argument("--random-crop", action='store_true', default=False,
                        help="random crop instead of center crop")
    parser.add_argument("--infomin", action='store_true', default=False,
                        help="use infomin augmentations")
    parser.add_argument("--epochs", type=int, default=100,
                        help='Number of epochs')
    parser.add_argument("--num-workers", type=int, default=10,
                        help='Number of CPUs')
    parser.add_argument("--seed", type=int, default=0,
                        help='seed')
    args = parser.parse_args()

    # Remove arguments that were not set and do not have default values
    args = {k: v for k, v in args.__dict__.items() if v is not None}
    return args


def main():
    root_dir = os.path.dirname(os.path.realpath(__file__))
    args_cmd = parse_args()
    with open(f'{root_dir}/yamls/{args_cmd["config"]}') as f:
        args = yaml.load(f, Loader)
    args.update(args_cmd)
    args = munchify(args)
    print(args.epochs)
    print(args_cmd)
    print()
    print(args)

    fix_random_seed(args.seed, benchmark=True, deterministic=True)

    aug = TwoCropsTransformInfoMin if args.infomin else TwoCropsTransform
    dataset = PatchUnlabeledDataset(args.data_root, aug(center_crop=not args.random_crop,
                                                        in_channels=args.in_channels,
                                                        crop_size=(args.crop_size, args.crop_size)),
                                    pid_range=(0, 100), norm=args.norm)

    gpu = torch.device(args.device)
    args.exp_dir = '/'.join([args.root_dir, args.exp_dir]) + args['exp_suffix']
    os.makedirs(args.exp_dir, exist_ok=True)
    stats_file = open('/'.join([args.exp_dir, "stats.txt"]), "a", buffering=1)
    # print(" ".join(sys.argv))
    print(args, file=stats_file)

    per_device_batch_size = args.batch_size // args.world_size
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=per_device_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    model = VICReg(args).cuda(gpu)
    if args.optimizer == 'lars':
        optimizer = LARS(
            model.parameters(),
            lr=0,
            weight_decay=args.weight_decay,
            weight_decay_filter=exclude_bias_and_norm,
            lars_adaptation_filter=exclude_bias_and_norm,
        )
        scheduler = None
    # optimizer = build_optimizer(args, model)
    else:
        optimizer = optim.AdamW(model.parameters(), weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.base_lr, epochs=args.epochs,
                                                        steps_per_epoch=len(loader),
                                                        pct_start=0.0, anneal_strategy='cos', cycle_momentum=True,
                                                        base_momentum=0.85,
                                                        max_momentum=0.95, div_factor=10.0,
                                                        final_div_factor=10000.0, three_phase=False,
                                                        last_epoch=-1, verbose=False)

    print('/'.join([args.exp_dir, "model.pth"]))
    if os.path.exists('/'.join([args.exp_dir, "model.pth"])):
        ckpt = torch.load('/'.join([args.exp_dir, "model.pth"]), map_location="cpu")
        start_epoch = ckpt["epoch"]
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
    else:
        start_epoch = 0
    print('Number of patches:', len(dataset))

    start_time = last_logging = time.time()
    for epoch in range(start_epoch, args.epochs):
        for step, (x, y) in enumerate(loader, start=epoch * len(loader)):
            if args.optimizer == 'lars':
                lr = adjust_learning_rate(args, optimizer, loader, step)
            else:
                lr = optimizer.param_groups[0]['lr']

            x = x.cuda(gpu, non_blocking=True)
            y = y.cuda(gpu, non_blocking=True)

            optimizer.zero_grad()
            loss = model.forward(x, y)
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()

            current_time = time.time()
            if current_time - last_logging > args.log_freq_time:
                stats = dict(
                    epoch=epoch,
                    step=step,
                    loss=loss.item(),
                    time=int(current_time - start_time),
                    lr=lr,
                )
                print(json.dumps(stats))
                print(json.dumps(stats), file=stats_file)
                last_logging = current_time
        state = dict(
            epoch=epoch + 1,
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
        )
        torch.save(state, '/'.join([args.exp_dir, "model.pth"]))

    torch.save(model.backbone.state_dict(), '/'.join([args.exp_dir, f"{args.arch}.pth"]))


if __name__ == '__main__':
    main()
