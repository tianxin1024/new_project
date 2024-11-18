import copy
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from einops import repeat

import ipdb

def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
        coord_x = -1+(2*i+1)/W
        coord_y = -1+(2*i+1)/H
        normalize to (-1, 1)
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

models = {}


def register(name):
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator


class Siren(nn.Module):
    """
        Siren activation
        https://arxiv.org/abs/2006.09661
    """

    def __init__(self, w0=30):
        """
            w0 comes from the end of section 3
            it should be 30 for the first layer
            and 1 for the rest
        """
        super().__init__()
        self.w0 = torch.tensor(w0)

    def forward(self, x):
        return torch.sin(self.w0 * x)

    def extra_repr(self):
        return "w0={}".format(self.w0)


def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            print('sine_init for Siren...')
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            print('first_layer_sine_init for Siren...')
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)


class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_list, act='sine'):
        super().__init__()
        # pdb.set_trace()
        if act is None:
            self.act = None
        elif act.lower() == 'relu':
            self.act = nn.ReLU() 
        elif act.lower() == 'gelu':
            self.act = nn.GELU()
        elif act.lower() == 'sine':
            self.act = Siren()
        else:
            assert False, f'activation {act} is not supported'
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            if self.act:
                layers.append(self.act)
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)
        if act is not None and act.lower() == 'sine':
            self.layers.apply(sine_init)
            self.layers[0].apply(first_layer_sine_init)

    def forward(self, x):
        # pdb.set_trace()
        shape = x.shape[:-1]
        x = self.layers(x.contiguous().view(-1, x.shape[-1]))
        return x.view(*shape, -1)

def make(model_spec, args=None, load_sd=False):
    if args is not None:
        model_args = copy.deepcopy(model_spec['args'])
        model_args.update(args)
    else:
        model_args = model_spec['args']

    model = models[model_spec['name']](**model_args)
    if load_sd:
        pretrained_dict = model_spec['sd']
        model_dict = model.state_dict()
  
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        # model.load_state_dict(model_spec['sd'])
    return model
 

class Implicit_Transformer_Up(nn.Module):
    def __init__(self):
        super(Implicit_Transformer_Up, self).__init__()
        self.embedding_q = None
        self.scale_token = True
        local_ensemble = False 
        self.imnet = MLP(4, 12, [256, 256, 256, 256], 'gelu')

        if local_ensemble:
            w = {
                'name': 'mlp',
                'args': {
                    'in_dim': 4,
                    'out_dim': 1,
                    'hidden_list': [256],
                    'act': 'gelu'
                }
            }
            self.Weight = models.make(w)

            score = {
                'name': 'mlp',
                'args': {
                    'in_dim': 2,
                    'out_dim': 1,
                    'hidden_list': [256],
                    'act': 'gelu'
                }
            }
            self.Score = models.make(score)

    def forward(self, input):
        feat = input

        h = input.shape[-2]
        w = input.shape[-1]

        coord = make_coord((h, w)).cuda()
        scale = torch.ones_like(coord)
        scale[:, 0] *= 1 / feat.shape[-2] # h
        scale[:, 1] *= 1 / feat.shape[-1] # w

        coord = coord.unsqueeze(0)
        scale = scale.unsqueeze(0)
        # K
        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        #V
        bs, q = coord.shape[:2]
        value = F.grid_sample(
            feat, coord.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)
        #K
        coord_k = F.grid_sample(
            feat_coord, coord.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)

        if self.embedding_q:
            Q = self.embedding_q(coord.contiguous().view(bs * q, -1))
            K = self.embedding_q(coord_k.contiguous().view(bs * q, -1))
            rel = Q - K
            
            rel[:, 0] *= feat.shape[-2]
            rel[:, 1] *= feat.shape[-1]
            inp = rel
            if self.scale_token:
                scale_ = scale.clone()
                scale_[:, :, 0] *= feat.shape[-2]
                scale_[:, :, 1] *= feat.shape[-1]
                # scale = scale.view(bs*q,-1)
                scale_ = self.embedding_s(scale_.contiguous().view(bs * q, -1))
                inp = torch.cat([inp, scale_], dim=-1)

        else:
            Q, K = coord, coord_k
            rel = Q - K
            rel[:, :, 0] *= feat.shape[-2]
            rel[:, :, 1] *= feat.shape[-1]
            inp = rel
            if self.scale_token:
                scale_ = scale.clone()
                scale_[:, :, 0] *= feat.shape[-2]
                scale_[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([inp, scale_], dim=-1)
        
        
        ipdb.set_trace()
        weight = self.imnet(inp.view(bs * q, -1)).view(bs * q, feat.shape[1], 3)
        pred = torch.bmm(value.contiguous().view(bs * q, 1, -1), weight).view(bs, q, -1)
        ret = pred
 # 
 #        v_lst = [(i,j) for i in range(-1, 2, 2) for j in range(-1, 2, 2)]
 #        eps_shift = 1e-6
 #        preds = []
 #        for v in v_lst:
 #            vx = v[0]
 #            vy = v[1]
 #            # project to LR field 
 #            ipdb.set_trace()
 #            tx = ((feat.shape[-2] - 1) / (1 - scale[:,0,0])).view(feat.shape[0],  1)
 #            ty = ((feat.shape[-1] - 1) / (1 - scale[:,0,1])).view(feat.shape[0],  1)
 #            rx = (2*abs(vx) -1) / tx if vx != 0 else 0
 #            ry = (2*abs(vy) -1) / ty if vy != 0 else 0
 #            bs, q = coord.shape[:2]
 #            coord_ = coord.clone()

 #            if vx != 0:
 #                coord_[:, :, 0] += vx /abs(vx) * rx + eps_shift
 #            if vy != 0:
 #                coord_[:, :, 1] += vy /abs(vy) * ry + eps_shift
 #            coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
 #            #Interpolate K to HR resolution  
 #            value = F.grid_sample(
 #                feat, coord_.flip(-1).unsqueeze(1),
 #                mode='nearest', align_corners=False)[:, :, 0, :] \
 #                .permute(0, 2, 1)
 #            #Interpolate K to HR resolution 
 #            coord_k = F.grid_sample(
 #                feat_coord, coord_.flip(-1).unsqueeze(1),
 #                mode='nearest', align_corners=False)[:, :, 0, :] \
 #                .permute(0, 2, 1)

 #            #calculate relation of Q-K
 #            if self.embedding_q:
 #                Q = self.embedding_q(coord.contiguous().view(bs * q, -1))
 #                K = self.embedding_q(coord_k.contiguous().view(bs * q, -1))
 #                rel = Q - K
 #                
 #                rel[:, 0] *= feat.shape[-2]
 #                rel[:, 1] *= feat.shape[-1]
 #                inp = rel
 #                if self.scale_token:
 #                    scale_ = scale.clone()
 #                    scale_[:, :, 0] *= feat.shape[-2]
 #                    scale_[:, :, 1] *= feat.shape[-1]
 #                    # scale = scale.view(bs*q,-1)
 #                    scale_ = self.embedding_s(scale_.contiguous().view(bs * q, -1))
 #                    inp = torch.cat([inp, scale_], dim=-1)
 #            else:
 #                Q, K = coord, coord_k
 #                rel = Q - K
 #                rel[:, :, 0] *= feat.shape[-2]
 #                rel[:, :, 1] *= feat.shape[-1]
 #                inp = rel
 #                if self.scale_token:
 #                    scale_ = scale.clone()
 #                    scale_[:, :, 0] *= feat.shape[-2]
 #                    scale_[:, :, 1] *= feat.shape[-1]
 #                    inp = torch.cat([inp, scale_], dim=-1)

 #            score = repeat(self.Score(rel.view(bs * q, -1)).view(bs, q, -1),'b q c -> b q (repeat c)', repeat=3)
 #            
 #            weight = self.imnet(inp.view(bs * q, -1)).view(bs * q, feat.shape[1], 3)
 #            pred = torch.bmm(value.contiguous().view(bs * q, 1, -1), weight).view(bs, q, -1)
 #            
 #            pred +=score
 #            preds.append(pred)

 #        preds = torch.stack(preds,dim=-1)

 #        ret = self.Weight(preds.view(bs*q*3, -1)).view(bs, q, -1)


        return ret





if __name__ == "__main__":
    input = torch.rand(1, 4, 80, 80).to('cuda')
    model = Implicit_Transformer_Up().to('cuda')

    output = model(input)
    print(output.shape)
