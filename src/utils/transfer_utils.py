import os 
from robustness import datasets, model_utils
from torchvision import models
from torchvision.datasets import CIFAR100
import torch as ch

from . import constants as cs
from . import fine_tunify
from .custom_models.vision_transformer import *

pytorch_models = {
    'alexnet': models.alexnet,
    'vgg16': models.vgg16,
    'vgg16_bn': models.vgg16_bn,
    'squeezenet': models.squeezenet1_0,
    'densenet': models.densenet161,
    'shufflenet': models.shufflenet_v2_x1_0,
    'mobilenet': models.mobilenet_v2,
    'resnext50_32x4d': models.resnext50_32x4d,
    'mnasnet': models.mnasnet1_0
}

vitmodeldict = {
    # ImageNet
    'deit_tiny_patch16_224': deit_tiny_patch16_224,
    'deit_small_patch16_224': deit_small_patch16_224,
    'deit_base_patch16_224': deit_base_patch16_224,
    'deit_base_patch16_384': deit_base_patch16_384,
    ##CIFAR10
    'deit_tiny_patch4_32': deit_tiny_patch4_32,
    'deit_small_patch4_32': deit_small_patch4_32,
    'deit_base_patch4_32': deit_base_patch4_32,
}

TRANSFER_DATASETS = ['cifar10', 'cifar100']

def get_dataset_and_loaders(args, shuffle_train=True, shuffle_val=False):
    '''Given arguments, returns a datasets object and the train and validation loaders.
    '''
    if args.dataset in ['imagenet', 'stylized_imagenet']:
        ds = datasets.ImageNet(args.data)
        img_size = 224
    elif args.dataset == 'cifar10':
        ds = datasets.CIFAR(args.data)
        ds.transform_train = cs.TRAIN_TRANSFORMS
        ds.transform_test = cs.TEST_TRANSFORMS
        img_size = 32
    elif args.dataset == 'cifar100':
        ds = datasets.CIFAR(args.data, num_classes=100, name='cifar100',
                    mean=[0.5071, 0.4867, 0.4408], 
                    std=[0.2675, 0.2565, 0.2761])
        ds.transform_train = cs.TRAIN_TRANSFORMS
        ds.transform_test = cs.TEST_TRANSFORMS
        ds.custom_class = CIFAR100
        img_size = 32

    train_loader, val_loader = ds.make_loaders(only_val=args.eval_only, batch_size=args.batch_size, 
                                    workers=args.workers, shuffle_train=shuffle_train, shuffle_val=shuffle_val)
    return ds, train_loader, val_loader


def resume_finetuning_from_checkpoint(args, ds, finetuned_model_path):
    '''Given arguments, dataset object and a finetuned model_path, returns a model
    with loaded weights and returns the checkpoint necessary for resuming training.
    '''
    print('[Resuming finetuning from a checkpoint...]')
    arch, add_custom_forward = get_arch(args)
    if args.dataset in TRANSFER_DATASETS:
        model, _ = model_utils.make_and_restore_model(
            arch=arch, dataset=datasets.ImageNet(''), add_custom_forward=add_custom_forward)
        while hasattr(model, 'model'):
            model = model.model
        model = fine_tunify.ft(
            args.arch, model, ds.num_classes, args.additional_hidden)
        model, checkpoint = model_utils.make_and_restore_model(arch=model, dataset=ds, resume_path=finetuned_model_path,
                                                               add_custom_forward=args.additional_hidden > 0 or add_custom_forward)
    else:
        model, checkpoint = model_utils.make_and_restore_model(
            arch=arch, dataset=ds, resume_path=finetuned_model_path,
            add_custom_forward=add_custom_forward)
    return model, checkpoint


def get_arch(args):
    add_custom_forward = True
    if args.arch in pytorch_models.keys():
        arch = pytorch_models[args.arch](args.pytorch_pretrained)
    elif args.arch in vitmodeldict:
        arch = vitmodeldict[args.arch](pretrained=args.pytorch_pretrained,
                                        num_classes=1000,
                                        drop_rate=0.,
                                        drop_path_rate=0.1)
    else:
        arch = args.arch
        add_custom_forward = False
    return arch, add_custom_forward

def get_model(args, ds):
    '''Given arguments and a dataset object, returns an ImageNet model (with appropriate last layer changes to 
    fit the target dataset) and a checkpoint. The checkpoint is set to None if not resuming training.
    '''
    finetuned_model_path = os.path.join(
        args.out_dir, args.exp_name, args.resume_ckpt_name)

    if args.resume and os.path.isfile(finetuned_model_path):
        # fix hijacking of normalizer
        patch_state_dict(finetuned_model_path)
        model, checkpoint = resume_finetuning_from_checkpoint(
            args, ds, finetuned_model_path)
    else:
        arch, add_custom_forward = get_arch(args)
        if args.dataset in TRANSFER_DATASETS:
            model, _ = model_utils.make_and_restore_model(
                arch=arch,
                dataset=datasets.ImageNet(''), resume_path=args.model_path, pytorch_pretrained=args.pytorch_pretrained,
                add_custom_forward=add_custom_forward)
            checkpoint = None
        else:
            model, _ = model_utils.make_and_restore_model(arch=arch, dataset=ds,
                                                resume_path=args.model_path, pytorch_pretrained=args.pytorch_pretrained,
                                                add_custom_forward=add_custom_forward)
            checkpoint = None

        if not args.no_replace_last_layer and not args.eval_only and args.dataset in TRANSFER_DATASETS:
            print(f'[Replacing the last layer with {args.additional_hidden} '
                  f'hidden layers and 1 classification layer that fits the {args.dataset} dataset.]')
            while hasattr(model, 'model'):
                model = model.model
            model = fine_tunify.ft(
                args.arch, model, ds.num_classes, args.additional_hidden)
            model, checkpoint = model_utils.make_and_restore_model(arch=model, dataset=ds,
                                                                   add_custom_forward=args.additional_hidden > 0 or add_custom_forward)
        else:
            print('[NOT replacing the last layer]')
    return model, checkpoint


def freeze_model(model, freeze_level):
    '''
    Freezes up to args.freeze_level layers of the model (assumes a resnet model)
    '''
    # Freeze layers according to args.freeze-level
    update_params = None
    if freeze_level != -1:
        # assumes a resnet architecture
        assert len([name for name, _ in list(model.named_parameters())
                    if f"layer{freeze_level}" in name]), "unknown freeze level (only {1,2,3,4} for ResNets)"
        update_params = []
        freeze = True
        for name, param in model.named_parameters():
            print(name, param.size())

            if not freeze and f'layer{freeze_level}' not in name:
                print(f"[Appending the params of {name} to the update list]")
                update_params.append(param)
            else:
                param.requires_grad = False

            if freeze and f'layer{freeze_level}' in name:
                # if the freeze level is detected stop freezing onwards
                freeze = False
    return update_params

def patch_state_dict(path): 
    pth = torch.load(path)
    d = pth['model']

    if ("normalizer.1.new_mean" in d or "normalizer.1.new_std" in d
        or "module.normalizer.1.new_mean" in d 
        or "module.normalizer.1.new_std" in d
        or "normalizer.normalizer.new_mean" in d 
        or "normalizer.normalizer.new_std" in d
        or "module.normalizer.normalizer.new_mean" in d 
        or "module.normalizer.normalizer.new_std" in d): 
        print("Patching normalizer module")
        new_d = {}
        for k in d: 
            new_k = k
            if k == "normalizer.1.new_mean": 
                new_k = "normalizer.new_mean"
            if k == "normalizer.1.new_std": 
                new_k = "normalizer.new_std"
            if k == "module.normalizer.1.new_mean": 
                new_k = "module.normalizer.new_mean"
            if k == "module.normalizer.1.new_std": 
                new_k = "module.normalizer.new_std"
            if k == "normalizer.normalizer.new_mean": 
                new_k = "normalizer.new_mean"
            if k == "normalizer.normalizer.new_std": 
                new_k = "normalizer.new_std"
            if k == "module.normalizer.normalizer.new_mean": 
                new_k = "module.normalizer.new_mean"
            if k == "module.normalizer.normalizer.new_std": 
                new_k = "module.normalizer.new_std"
            new_d[new_k] = d[k]
        pth['model'] = new_d
        torch.save(pth,path)
    return