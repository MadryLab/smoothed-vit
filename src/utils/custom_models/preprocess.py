import torch.nn as nn
import torch
import torch as ch

class StripeAblator(nn.Module):
    def __init__(self, ablation_size, dim=3):
        super().__init__()
        self.ablation_size = ablation_size
        self.dim = dim
            
    def forward(self, x, pos):
        k = self.ablation_size
        dim = self.dim
        total_pos = x.shape[dim]
        if pos + k > total_pos: 
            idx = [slice(None,None,None) if _ != dim else slice(pos+k-total_pos,pos,None) for _ in range(4)]
            x[idx] = 0
        else: 
            left_idx = [slice(None,None,None) if _ != dim else slice(0, pos, None) for _ in range(4)]
            right_idx = [slice(None,None,None) if _ != dim else slice(pos+k, total_pos, None) for _ in range(4)]
            x[left_idx] = 0
            x[right_idx] = 0
        return x

class BlockAblator(nn.Module):
    def __init__(self, ablation_size):
        super().__init__()
        self.ablation_size = ablation_size
            
    def forward(self, x, pos):
        """
        x: input to be ablated
        pos: tuple (idx_x, idx_y) representing the position of ablation to be applied

        returns: ablated image
        """
        assert len(pos) == 2

        k = self.ablation_size
        total_pos = x.shape[-1]
        pos_x, pos_y = pos
        x_orig = x.clone()
        x[:, :, pos_x:(pos_x + k), pos_y:(pos_y + k)] = 0
        if pos_x + k > total_pos and pos_y + k > total_pos:
            x[:, :, 0:(pos_x + k)%total_pos, 0:(pos_y + k)%total_pos] = 0
            x[:, :, 0:(pos_x + k)%total_pos, pos_y:(pos_y + k)] = 0
            x[:, :, pos_x:(pos_x + k), 0:(pos_y + k)%total_pos] = 0
        elif pos_x + k > total_pos:
            x[:, :, 0:(pos_x + k)%total_pos, pos_y:(pos_y + k)] = 0
        elif pos_y + k > total_pos:
            x[:, :, pos_x:(pos_x + k), 0:(pos_y + k)%total_pos] = 0

        return x_orig - x

class Simple224Upsample(nn.Module):
    # go from 32 to 224
    def __init__(self, arch=''):
        super(Simple224Upsample, self).__init__()
        self.upsample = nn.Upsample(mode='nearest', scale_factor=7)
        self.arch = arch
        
    def forward(self, x):
        return self.upsample(x)

class Upsample384AndPad(nn.Module):
    def __init__(self):
        super(Upsample384AndPad, self).__init__()
        self.upsample = nn.Upsample(mode='nearest', scale_factor=8) # 256
        self.zero_pad = torch.nn.ZeroPad2d((384-256)//2) # 64 on each side
    
    def forward(self, x, ones_mask):
        x = self.upsample(x)
        x = self.zero_pad(x)
        return x
    
cifar_upsamples = {
    'simple224': Simple224Upsample,
    'upsample384': Upsample384AndPad,
    'none': None,
}

class MaskProcessor(nn.Module):
    def __init__(self, patch_size=16):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(patch_size)
    
    def forward(self, ones_mask):
        B = ones_mask.shape[0]
        ones_mask = ones_mask[0].unsqueeze(0) # take the first mask
        ones_mask = self.avg_pool(ones_mask)[0]
        ones_mask = torch.where(ones_mask.view(-1) > 0)[0] + 1
        ones_mask = torch.cat([torch.cuda.IntTensor(1).fill_(0), ones_mask]).unsqueeze(0)
        ones_mask = ones_mask.expand(B, -1)
        return ones_mask
    
class PreProcessor(nn.Module):
    def __init__(self, normalizer, ablation_size, upsample_type='none',
                 return_mask=False, do_ablation=True, ablation_type='col', ablation_target=None):
        '''
        normalizer: the normalizer module
        ablation_size: size of ablation
        upsample_type: type of upsample (none, simple224, upsample384)
        return_mask: if true, keep the mask as a fourth channel
        do_ablation: perform the ablation
        ablation_target: the column to ablate. if None, pick a random column
        '''
        super().__init__()
        print({
            "ablation_size": ablation_size,
            "upsample_type": upsample_type,
            "return_mask": return_mask,
            "do_ablation": do_ablation,
            "ablation_target": ablation_target
        })
        if ablation_type == 'col':
            self.ablator = StripeAblator(ablation_size, dim=3)
        elif ablation_type == 'block':
            self.ablator = BlockAblator(ablation_size)
        else:
            raise Exception('Unkown ablation type')

        if upsample_type == 'none':
            self.upsampler = None
        else:
            self.upsampler = cifar_upsamples[upsample_type]()
        self.return_mask = return_mask
        self.normalizer = normalizer
        self.do_ablation = do_ablation
        self.ablation_target = ablation_target
    
    def forward(self, x):
        B, C, H, W = x.shape
        if C == 3:
            # we don't have a mask yet!!
            ones = torch.ones((B, 1, H, W)).cuda()
            x = torch.cat([x, ones], dim=1)
        else:
            assert not self.do_ablation, "cannot do ablation if already passed in ablation mask"
        if self.do_ablation:
            pos = self.ablation_target
            if pos is None:
                if isinstance(self.ablator, StripeAblator):
                    pos = ch.randint(x.shape[3], (1,))
                elif isinstance(self.ablator, BlockAblator):
                    pos = ch.randint(x.shape[3], (2,))
            x = self.ablator(x=x, pos=pos)
        if self.upsampler is not None:
            x = self.upsampler(x)
        x[:, :3] = self.normalizer(x[:, :3]) # normalize
        if self.return_mask:
            return x # WARNING returning 4 channel output
        else:
            return x[:, :3]
    
