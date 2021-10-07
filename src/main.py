import argparse
import os
import sys
from datetime import datetime

import cox.store
import numpy as np
import torch as ch
from robustness import datasets, defaults, train

from utils.transfer_utils import get_dataset_and_loaders, freeze_model, get_model, TRANSFER_DATASETS
from utils.custom_models.preprocess import PreProcessor
from utils.smoothing import certify

if int(os.environ.get("NOTEBOOK_MODE", 0)) == 1:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm as tqdm

parser = argparse.ArgumentParser(description='Transfer learning via pretrained Imagenet models',
                                 conflict_handler='resolve')
parser = defaults.add_args_to_parser(defaults.CONFIG_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.MODEL_LOADER_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.TRAINING_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.PGD_ARGS, parser)

# Custom arguments
parser.add_argument('--dataset', type=str, default='cifar',
                    help='Downstream task dataset (Overrides the one in robustness.defaults)')
parser.add_argument('--model-path', type=str, help='Path to model trained with robustness lib')
parser.add_argument('--resume', action='store_true',
                    help='Whether to resume or not (Overrides the one in robustness.defaults)')
parser.add_argument('--resume-ckpt-name', type=str, default='checkpoint.pt.latest', 
                    help='Name of the checkpoint to resume from')
parser.add_argument('--pytorch-pretrained', action='store_true',
                    help='If True, loads a Pytorch pretrained model.')
parser.add_argument('--subset', type=int, default=None,
                    help='number of training data to use from the dataset')
parser.add_argument('--no-tqdm', type=int, default=0,
                    choices=[0, 1], help='Do not use tqdm.')
parser.add_argument('--no-replace-last-layer', action='store_true',
                    help='Whether to avoid replacing the last layer')
parser.add_argument('--freeze-level', type=int, default=-1,
                    help='Up to what layer to freeze in the pretrained model (assumes a resnet architectures)')
parser.add_argument('--additional-hidden', type=int, default=0,
                    help='How many hidden layers to add on top of pretrained network + classification layer')
parser.add_argument('--update-BN-stats', action='store_true')


parser.add_argument('--cifar-preprocess-type', type=str, default='none', 
                    help='What cifar preprocess type to use (only use with CIFAR-10)',
                    choices=['simple224', 'upsample384', 'none'])
parser.add_argument('--drop-tokens', action='store_true', help='drop masked out tokens (for ViTs only)')
parser.add_argument('--ablation-target', type=int, default=None, help='choose specific column to keep')


## Input Ablation
parser.add_argument('--ablate-input', action='store_true')
parser.add_argument('--ablation-type', type=str, default='col',
                    help='Type of ablations', choices=['col', 'block'])
parser.add_argument('--ablation-size', type=int, default=4,
                    help='Width of the remaining column if --ablation-type is "col".' 
                    'Side length of the remaining block if --ablation-type is "block"')

# certification arguments
parser.add_argument('--skip-store', action='store_true')
parser.add_argument('--certify', action='store_true')
parser.add_argument('--certify-out-dir', default='OUTDIR_CERT')
parser.add_argument('--certify-mode', default='both', choices=['both', 'row', 'col', 'block'])
parser.add_argument('--certify-ablation-size', type=int, default=4)
parser.add_argument('--certify-patch-size', type=int, default=5)
parser.add_argument('--certify-stride', type=int, default=1)
parser.add_argument('--batch-id', type=int, default=None)


def main(args, store):
    '''Given arguments and a cox store, trains as a model. Check out the 
    argparse object in this file for argument options.
    '''
    ds, train_loader, validation_loader = get_dataset_and_loaders(args)

    model, checkpoint = get_model(args, ds)

    def get_n_params(model):
        total = 0
        for p in list(model.parameters()):
            total += np.prod(p.size())
        return total
    print(f'==> [Number of parameters of the model is {get_n_params(model)}]')
    
    model.normalizer = PreProcessor(
        normalizer=model.normalizer,
        ablation_size=args.ablation_size, 
        upsample_type=args.cifar_preprocess_type,
        return_mask=args.drop_tokens,
        do_ablation=args.ablate_input,
        ablation_type=args.ablation_type,
        ablation_target=args.ablation_target,
    )
    if 'deit' not in args.arch:
        assert args.drop_tokens == False
    
    if args.update_BN_stats:
        print(f'==>[Started updating the BN stats relevant to {args.dataset}]')
        assert not hasattr(model, "module"), "model is already in DataParallel."
        model = model.cuda()
        model.train()
        attack_kwargs = {
            'constraint': args.constraint,
            'eps': args.eps,
            'step_size': args.attack_lr,
            'iterations': args.attack_steps,
        }
        with ch.no_grad():
            niters = 0
            while niters < 200:
                iterator = tqdm(enumerate(train_loader), total=len(train_loader))
                for _, (inp, _) in iterator:
                    model(inp.cuda(), **attack_kwargs)
                    niters += 1
        print('==>[Updated the BN stats]')

    if args.eval_only:
        if args.certify: 
            return certify(args, model, validation_loader, store=store)
        else: 
            return train.eval_model(args, model, validation_loader, store=store)

    update_params = freeze_model(model, freeze_level=args.freeze_level)
    print(f"Dataset: {args.dataset} | Model: {args.arch}")

    train.train_model(args, model, (train_loader, validation_loader), store=store,
                        checkpoint=checkpoint, update_params=update_params)


def args_preprocess(args):
    '''
    Fill the args object with reasonable defaults, and also perform a sanity check to make sure no
    args are missing.
    '''
    if args.adv_train and eval(args.eps) == 0:
        print('[Switching to standard training since eps = 0]')
        args.adv_train = 0

    if args.pytorch_pretrained:
        assert not args.model_path, 'You can either specify pytorch_pretrained or model_path, not together.'


    ALL_DS = TRANSFER_DATASETS + ['imagenet', 'stylized_imagenet']
    assert args.dataset in ALL_DS

    # Important for automatic job retries on the cluster in case of premptions. Avoid uuids.
    assert args.exp_name != None

    # Preprocess args
    default_ds = args.dataset if args.dataset in datasets.DATASETS else "cifar"
    args = defaults.check_and_fill_args(args, defaults.CONFIG_ARGS, datasets.DATASETS[default_ds])
    if not args.eval_only:
        args = defaults.check_and_fill_args(args, defaults.TRAINING_ARGS, datasets.DATASETS[default_ds])
    if args.adv_train or args.adv_eval:
        args = defaults.check_and_fill_args(args, defaults.PGD_ARGS, datasets.DATASETS[default_ds])
    args = defaults.check_and_fill_args(args, defaults.MODEL_LOADER_ARGS, datasets.DATASETS[default_ds])

    return args


if __name__ == "__main__":
    args = parser.parse_args()
    args = args_preprocess(args)

    # Create store and log the args
    if args.skip_store: 
        store = None
    else:
        store = cox.store.Store(args.out_dir, args.exp_name)
        if 'metadata' not in store.keys:
            args_dict = args.__dict__
            schema = cox.store.schema_from_dict(args_dict)
            store.add_table('metadata', schema)
            store['metadata'].append_row(args_dict)
        else:
            print('[Found existing metadata in store. Skipping this part.]')

        ## Save python command to a file
        cmd = 'python ' + ' '.join(sys.argv)
        now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        file_path = os.path.join(args.out_dir, args.exp_name, 'command.sh')
        with open(file_path, 'a') as f:
            f.write(now + '\n')
            f.write(cmd + '\n')

    main(args, store)
