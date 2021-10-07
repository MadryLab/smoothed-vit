import torch as ch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import os
import itertools
import math

def ablate(x, pos, k, total_pos, dim): 
    # x : input
    # pos : starting position
    # k : size of ablation
    # total_pos : maximum position
    # dim : height or width (2 or 3)
    inp = ch.zeros_like(x)
    mask = x.new_zeros(x.size(0), 1, x.size(2), x.size(3))
    if pos + k > total_pos: 
        idx1 = [slice(None,None,None) if _ != dim else slice(pos,total_pos,None) for _ in range(4)]
        idx2 = [slice(None,None,None) if _ != dim else slice(0,pos+k-total_pos,None) for _ in range(4)]
        inp[idx1] = x[idx1]
        inp[idx2] = x[idx2]
        mask[idx1] = 1
        mask[idx2] = 1
    else: 
        idx = [slice(None,None,None) if _ != dim else slice(pos,pos+k,None) for _ in range(4)]
        inp[idx] = x[idx]
        mask[idx] = 1
    return ch.cat([inp,mask],dim=1)

def ablate2(x,block_pos,block_k,shape): 
    inp = ch.zeros_like(x)
    mask = x.new_zeros(x.size(0), 1, x.size(2), x.size(3))

    slices = []
    for pos,k,total_pos in zip(block_pos,block_k,shape): 
        if pos + k > total_pos: 
            slices.append([slice(0,pos+k-total_pos,None), slice(pos,total_pos,None)])
        else: 
            slices.append([slice(pos,pos+k,None)])

    for si,sj in itertools.product(*slices): 
        idx = [slice(None,None,None),slice(None,None,None),si,sj]
        inp[idx] = x[idx]
        mask[idx] = 1

    return ch.cat([inp,mask],dim=1)
    

class DerandomizedSmoother(nn.Module): 
    def __init__(self, column_model=None, row_model=None, block_size=(4,4), stride=(1,1), preprocess=None): 
        super(DerandomizedSmoother, self).__init__()
        self.column_model = column_model
        self.row_model = row_model
        self.block_size = block_size
        self.stride = stride
        self.preprocess = preprocess
        
    def forward(self, x, nclasses=10, threshold=None, return_mode=None): 
        # return_mode == 'differentiable', 'ablations', 'predictions'
        nex, nch, h, w = x.size()
        
        predictions = x.new_zeros(nex, nclasses)
        softmaxes = 0
        ablations = []
        for model, total_pos, k, s, dim in zip((self.row_model, self.column_model), 
                                               (h,w), 
                                               self.block_size, 
                                               self.stride, 
                                               (2,3)): 
            if model is not None: 
                for pos in range(0,total_pos,s): 
                    inp = ablate(x, pos, k, total_pos, dim)
                    if self.preprocess is not None: 
                        inp = self.preprocess(inp)
                    out = model(inp)
                    if isinstance(out, tuple): 
                        out = out[0]
                    out = F.softmax(out,dim=1)

                    if return_mode == 'differentiable': 
                        softmaxes += out
                    if return_mode == 'ablations' or return_mode == 'all': 
                        ablations.append(out.max(1)[1].unsqueeze(1))

                    if threshold is not None: 
                        predictions += (out >= threshold).int()
                    else: 
                        predictions += (out.max(1)[0].unsqueeze(1) == out).int()
        
        if return_mode == 'differentiable': 
            return softmaxes/len(range(0,total_pos,s))
        if return_mode == 'predictions': 
            return predictions.argmax(1), predictions
        if return_mode == 'ablations': 
            return predictions.argmax(1), ch.cat(ablations,dim=1)
        if return_mode == 'all': 
            return predictions.argmax(1), predictions, ch.cat(ablations,dim=1)

        return predictions.argmax(1)

class BlockDerandomizedSmoother(nn.Module): 
    def __init__(self, block_model=None, block_size=(4,4), stride=(1,1), preprocess=None): 
        super(BlockDerandomizedSmoother, self).__init__()
        self.model = block_model
        self.block_size = block_size
        self.stride = stride
        self.preprocess = preprocess
        
    def forward(self, x, nclasses=10, threshold=None, return_mode=None): 
        # return_mode == 'differentiable', 'ablations', 'predictions'
        nex, nch, h, w = x.size()
        
        predictions = x.new_zeros(nex, nclasses)
        softmaxes = 0
        ablations = []

        for i_pos in tqdm(range(0,h,self.stride[0])): 
            for j_pos in range(0,w,self.stride[1]): 
                inp = ablate2(x, (i_pos,j_pos), self.block_size, (h,w))
                if self.preprocess is not None: 
                    inp = self.preprocess(inp)
                out = self.model(inp)
                if isinstance(out, tuple): 
                    out = out[0]
                out = F.softmax(out,dim=1)

                if return_mode == 'differentiable': 
                    softmaxes += out
                if return_mode == 'ablations' or return_mode == 'all': 
                    ablations.append(out.max(1)[1].unsqueeze(1))

                if threshold is not None: 
                    predictions += (out >= threshold).int()
                else: 
                    predictions += (out.max(1)[0].unsqueeze(1) == out).int()
        
        if return_mode == 'differentiable': 
            return softmaxes/len(range(0,total_pos,s))
        if return_mode == 'predictions': 
            return predictions.argmax(1), predictions
        if return_mode == 'ablations': 
            return predictions.argmax(1), ch.cat(ablations,dim=1)
        if return_mode == 'all': 
            return predictions.argmax(1), predictions, ch.cat(ablations,dim=1)


        return predictions.argmax(1)

def certify(args, model, validation_loader, store=None): 
    # print("Certification is replacing transform with ToTensor")
    m = args.certify_patch_size
    s = args.certify_ablation_size
    stride = args.certify_stride

    if args.dataset == 'cifar10': 
        nclasses = 10
    elif args.dataset == 'imagenet': 
        nclasses = 1000
    else: 
        raise ValueError("Unknown number of classes")

    os.makedirs(args.certify_out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.certify_out_dir, args.exp_name), exist_ok=True)
    summary_path = os.path.join(args.certify_out_dir,args.exp_name,f"m{m}_s{s}_summary.pth")
    if os.path.exists(summary_path): 
        d = ch.load(summary_path)
        print("summary:")
        print(f"acc: {d['acc']:.4f}, abl {d['ablation_acc']:.4f},  cert {d['cert_acc']:.4f}, delta: {d['delta']:.4f}, s: {s}, m: {m}")
        return d['delta']

    model.eval() 
    model = nn.DataParallel(model)
    with ch.no_grad(): 
        col_model = model if args.certify_mode in ['both', 'col'] else None
        row_model = model if args.certify_mode in ['both', 'row'] else None

        # number of ablations in one axis
        na = math.ceil((m + s - 1)/stride)
        if args.certify_mode == 'block': 
            smoothed_model = BlockDerandomizedSmoother(
                block_model=model, 
                block_size=(s,s), 
                stride=(stride,stride)
            )
            gap = 2*(na**2) + 1
        else: 
            smoothed_model = DerandomizedSmoother(
                column_model=col_model, 
                row_model=row_model, 
                block_size=(s,s), 
                stride=(stride,stride)
            )
            # add one to not handle ties
            # 2*(m + s - 1) for one dimension of ablations, and 
            # double again for two axes
            factor = 4 if args.certify_mode == 'both' else 2
            gap = na*factor + 1 

        total = 0
        n = 0
        smooth_total = 0
        certified_total = 0
        ablation_total = 0
        delta = 0
        

        pbar = tqdm(validation_loader)
        for i,(X,y) in enumerate(pbar): 
            if args.batch_id != None and args.batch_id < i: 
                break
            if args.batch_id != None and args.batch_id != i: 
                continue
            file_path = os.path.join(args.certify_out_dir,args.exp_name,f"m{m}_s{s}_b{i}.pth")
            if os.path.exists(file_path): 
                d = ch.load(file_path)

                smooth_total += d['smooth_delta']
                ablation_total += d['ablation_delta']
                certified_total += d['certified_delta']
                total += d['total_delta']
                delta += d['delta_delta']
                n += X.size(0)

                pbar.set_description(f"Acc: {total/n:.4f} Abl acc: {ablation_total/n:.4f} Smo acc: {smooth_total/n:.4f} Cer acc: {certified_total/n:.4f} Delta: {delta/n:.0f}")
                continue
            X,y = X.cuda(),y.cuda()
            acc = (model(X)[0].max(1)[1] == y).float().mean()

            y_smoothed, y_counts, y_ablations = smoothed_model(X, return_mode="all", nclasses=nclasses)
            y_1st_vals, y_1st_idx = y_counts.kthvalue(nclasses,dim=1)
            y_2nd_vals, y_2nd_idx = y_counts.kthvalue(nclasses-1,dim=1)

            y_tar_vals = ch.gather(y_counts,1,y.unsqueeze(1)).squeeze()
            not_y = (y_1st_idx != y)
            y_nex_idx = y_1st_idx*(not_y.int()) + y_2nd_idx*(~not_y)
            y_nex_vals = ch.gather(y_counts,1,y_nex_idx.unsqueeze(1)).squeeze()
            
            y_certified = (y == y_1st_idx)*(y_1st_vals >= y_2nd_vals + gap)

            smooth_delta = (y_smoothed == y).sum().item()
            smooth_total += smooth_delta

            ablation_delta = y_tar_vals.sum().item()
            ablation_total += ablation_delta

            certified_delta = y_certified.sum().item()
            certified_total += certified_delta

            total_delta = acc.item()*X.size(0)
            total += total_delta

            delta_delta = (y_tar_vals - y_nex_vals).sum().item()
            delta += delta_delta
            n += X.size(0)

            ch.save({
                "total_delta" : total_delta, 
                "certified_delta" : certified_delta, 
                "smooth_delta": smooth_delta, 
                "ablation_delta": ablation_delta, 
                "delta_delta": delta_delta, 
                "s": s, 
                "m": m, 
                "mode": args.certify_mode, 
                "ablations": y_ablations.detach().cpu(), 
                "y": y.cpu()
            }, file_path)

            pbar.set_description(f"Acc: {total/n:.4f} Abl acc: {ablation_total/n:.4f} Smo acc: {smooth_total/n:.4f} Cer acc: {certified_total/n:.4f} Delta: {delta/n:.0f}")

        if args.batch_id == None: 
            ch.save({
                "acc" : total/n, 
                "cert_acc" : certified_total/n, 
                "smooth_acc": smooth_total/n, 
                "ablation_acc": ablation_total/n, 
                "delta": delta/n, 
                "s": s, 
                "m": m, 
                "mode": args.certify_mode
            }, summary_path)

            print(f"acc: {total/n:.4f}, ablation {ablation_total/n:.4f}, smoothed {smooth_total/n:.4f}, certified {certified_total/n:.4f}, delta: {delta/n:.4f}, s: {s}, m: {m}")
        return delta/n