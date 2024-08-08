'''Adapted from https://github.com/GT-RIPL/CODA-Prompt/blob/main/models/zoo.py
'''

import torch
import torch.nn as nn
import copy


SUPPORTED_CONFIGS = {
    'CodaPrompt': {
        'default': dict(
            n_tasks=36,
            e_pool_size=36 * 5,
            e_p_length=2,
            e_layers=[0,1,2,3,4]
        ),
    },
    'DualPrompt': {
        'default': dict(
            n_tasks=36,
            top_k=1,
            task_id_bootstrap=True,
            g_layers=[0,1],
            e_layers=[2,3,4],
            g_p_length=2,
            e_p_length=16,
            e_pool_size=36,
        ),
    }
}

for length in [2, 4, 6, 8]:
    for mul in [1, 3, 5, 7]:
        this_config = copy.deepcopy(SUPPORTED_CONFIGS['CodaPrompt']['default'])
        this_config['e_p_length'] = length
        this_config['e_pool_size'] = 36 * mul
        SUPPORTED_CONFIGS['CodaPrompt'][f'L{length}_S{mul}'] = this_config

for length1 in [2, 4, 6, 8]:
    for length2 in [2, 4, 6, 8, 16]:
        this_config = copy.deepcopy(SUPPORTED_CONFIGS['DualPrompt']['default'])
        this_config['g_p_length'] = length1
        this_config['e_p_length'] = length2
        SUPPORTED_CONFIGS['DualPrompt'][f'L{length1}_{length2}'] = this_config


# @inproceedings{smith2023CodaPrompt,
#   title = {CODA-Prompt: COntinual Decomposed Attention-Based Prompting for Rehearsal-Free Continual Learning},
#   booktitle = {Proceedings of the Conference on Computer Vision and Pattern Recognition},
#   author = {Smith, James Seale and Karlinsky, Leonid and Gutta, Vyshnavi and {Cascante-Bonilla}, Paola and Kim, Donghyun and Arbelle, Assaf and Panda, Rameswar and Feris, Rogerio and Kira, Zsolt},
#   year = {2023},
#   pages = {11909--11919}
# }
class CodaPrompt(nn.Module):
    def __init__(self, prompt_config=None, 
                 embed_dim=512, key_embed_dim=None, 
                 n_tasks=36, e_pool_size=36*5, e_p_length=8, e_layers=[0, 1, 2, 3, 4]):
        super().__init__()
        self.register_buffer('task_count', torch.LongTensor([-1]))

        if prompt_config is None:
            self.n_tasks = n_tasks
            self.e_pool_size = e_pool_size
            self.e_p_length = e_p_length
            self.e_layers = e_layers
        else:
            for k, v in SUPPORTED_CONFIGS['CodaPrompt'][prompt_config].items():
                setattr(self, k, v)

        self.embed_dim = embed_dim
        self.key_embed_dim = key_embed_dim or embed_dim
        
        # e prompt init
        for e in self.e_layers:
            e_l = self.e_p_length
            p = tensor_prompt(self.e_pool_size, e_l, embed_dim)
            k = tensor_prompt(self.e_pool_size, self.key_embed_dim)
            a = tensor_prompt(self.e_pool_size, self.key_embed_dim)
            p = self.gram_schmidt(p)
            k = self.gram_schmidt(k)
            a = self.gram_schmidt(a)
            setattr(self, f'e_p_{e}', p)
            setattr(self, f'e_k_{e}', k)
            setattr(self, f'e_a_{e}', a)

    def process_task_count(self):
        self.task_count += 1
        assert self.task_count < self.n_tasks

        for e in self.e_layers:
            K = getattr(self,f'e_k_{e}')
            A = getattr(self,f'e_a_{e}')
            P = getattr(self,f'e_p_{e}')
            k = self.gram_schmidt(K)
            a = self.gram_schmidt(A)
            p = self.gram_schmidt(P)
            setattr(self, f'e_p_{e}',p)
            setattr(self, f'e_k_{e}',k)
            setattr(self, f'e_a_{e}',a)

    # code for this function is modified from:
    # https://github.com/legendongary/pytorch-gram-schmidt/blob/master/gram_schmidt.py
    def gram_schmidt(self, vv):
        def projection(u, v):
            return (v * u).sum() / (u * u).sum() * u

        # check if the tensor is 3D and flatten the last two dimensions if necessary
        is_3d = len(vv.shape) == 3
        if is_3d:
            shape_2d = copy.deepcopy(vv.shape)
            vv = vv.view(vv.shape[0],-1)

        # swap rows and columns
        vv = vv.T

        # process matrix size
        nk = vv.size(1)
        uu = torch.zeros_like(vv, device=vv.device)

        # get starting point
        pt = int(self.e_pool_size / (self.n_tasks))
        s = int(self.task_count * pt)
        f = int((self.task_count + 1) * pt)
        if s > 0:
            uu[:, 0:s] = vv[:, 0:s].clone()
        for k in range(s, f):
            redo = True
            while redo:
                redo = False
                vk = torch.randn_like(vv[:,k]).to(vv.device)
                uk = 0
                for j in range(0, k):
                    if not redo:
                        uj = uu[:, j].clone()
                        proj = projection(uj, vk)
                        if proj is None:
                            redo = True
                            print('restarting!!!')
                        else:
                            uk = uk + proj
                if not redo: uu[:, k] = vk - uk
        for k in range(s, f):
            uk = uu[:, k].clone()
            uu[:, k] = uk / (uk.norm())

        # undo swapping of rows and columns
        uu = uu.T 

        # return from 2D
        if is_3d:
            uu = uu.view(shape_2d)
        
        return torch.nn.Parameter(uu) 

    def forward(self, x_querry):
        prompt_keys, prompt_values = [], []
        for l in self.e_layers:
            K = getattr(self,f'e_k_{l}')
            A = getattr(self,f'e_a_{l}')
            p = getattr(self,f'e_p_{l}')
            pt = int(self.e_pool_size / (self.n_tasks))
            s = int(self.task_count * pt)
            f = int((self.task_count + 1) * pt)
            
            # freeze/control past tasks
            if self.training:
                if self.task_count > 0:
                    K = torch.cat((K[:s].detach().clone(),K[s:f]), dim=0)
                    A = torch.cat((A[:s].detach().clone(),A[s:f]), dim=0)
                    p = torch.cat((p[:s].detach().clone(),p[s:f]), dim=0)
                else:
                    K = K[s:f]
                    A = A[s:f]
                    p = p[s:f]
            else:
                K = K[0:f]
                A = A[0:f]
                p = p[0:f]

            # with attention and cosine sim
            # (b x 1 x d) * soft([1 x k x d]) = (b x k x d) -> attention = k x d
            a_querry = torch.einsum('bd,kd->bkd', x_querry, A)
            # # (b x k x d) - [1 x k x d] = (b x k) -> key = k x d
            n_K = nn.functional.normalize(K, dim=1)
            q = nn.functional.normalize(a_querry, dim=2)
            aq_k = torch.einsum('bkd,kd->bk', q, n_K)
            # (b x 1 x k x 1) * [1 x plen x k x d] = (b x plen x d) -> prompt = plen x k x d
            P_ = torch.einsum('bk,kld->bld', aq_k, p)

            # select prompts
            i = int(self.e_p_length/2)
            Ek = P_[:,:i,:]
            Ev = P_[:,i:,:]

            prompt_keys.append(Ek)
            prompt_values.append(Ev)

        prompt_keys = torch.stack(prompt_keys, dim=0)
        prompt_values = torch.stack(prompt_values, dim=0)
        loss = 0
        return (prompt_keys, prompt_values), loss


# @inproceedings{wang2022DualPrompt,
#   title = {DualPrompt: Complementary Prompting for~Rehearsal-Free Continual Learning},
#   booktitle = {Proceedings of the European Conference on Computer Vision},
#   author = {Wang, Zifeng and Zhang, Zizhao and Ebrahimi, Sayna and Sun, Ruoxi and Zhang, Han and Lee, Chen-Yu and Ren, Xiaoqi and Su, Guolong and Perot, Vincent and Dy, Jennifer and Pfister, Tomas},
#   year = {2022},
#   pages = {631--648},
#   doi = {10.1007/978-3-031-19809-0_36}
# }
class DualPrompt(nn.Module):
    def __init__(self, prompt_config=None, 
                 embed_dim=512, key_embed_dim=None, 
                 n_tasks=36, top_k=1, task_id_bootstrap=True, 
                 g_layers=[0,1], e_layers=[2,3,4], 
                 g_p_length=6, e_p_length=20, e_pool_size=36):
        super().__init__()
        self.register_buffer('task_count', torch.LongTensor([-1]))

        if prompt_config is None:
            self.n_tasks = n_tasks
            self.top_k = top_k
            self.task_id_bootstrap = task_id_bootstrap
            self.g_layers = g_layers
            self.e_layers = e_layers
            self.g_p_length = g_p_length
            self.e_p_length = e_p_length
            self.e_pool_size = e_pool_size
        else:
            for k, v in SUPPORTED_CONFIGS['DualPrompt'][prompt_config].items():
                setattr(self, k, v)

        self.embed_dim = embed_dim
        self.key_embed_dim = key_embed_dim or embed_dim
        
        # g prompt init
        for g in self.g_layers:
            p = tensor_prompt(self.g_p_length, embed_dim)
            setattr(self, f'g_p_{g}', p)

        # e prompt init
        for e in self.e_layers:
            p = tensor_prompt(self.e_pool_size, self.e_p_length, embed_dim)
            k = tensor_prompt(self.e_pool_size, self.key_embed_dim)
            setattr(self, f'e_p_{e}', p)
            setattr(self, f'e_k_{e}', k)

    def process_task_count(self):
        self.task_count += 1

    def forward(self, x_querry):

        all_layers = self.g_layers + self.e_layers
        assert min(all_layers) == 0

        prompt_keys, prompt_values = [], []
        total_loss = 0
        for l in range(max(all_layers)):
            loss = 0

            # e prompts
            e_valid = False
            if l in self.e_layers:
                e_valid = True
                B, C = x_querry.shape
                K = getattr(self,f'e_k_{l}') # 0 based indexing here
                p = getattr(self,f'e_p_{l}') # 0 based indexing here
                
                # cosine similarity to match keys/querries
                n_K = nn.functional.normalize(K, dim=1)
                q = nn.functional.normalize(x_querry, dim=1).detach()
                cos_sim = torch.einsum('bj,kj->bk', q, n_K)
                
                if self.training:
                    # dual prompt during training uses task id
                    if self.task_id_bootstrap:
                        loss = (1.0 - cos_sim[:,self.task_count]).mean()
                        P_ = p[self.task_count].expand(len(x_querry),-1,-1)
                    else:
                        top_k = torch.topk(cos_sim, self.top_k, dim=1)
                        k_idx = top_k.indices
                        loss = (1.0 - cos_sim[:,k_idx]).mean()
                        P_ = p[k_idx]
                else:
                    top_k = torch.topk(cos_sim, self.top_k, dim=1)
                    k_idx = top_k.indices
                    P_ = p[k_idx]
                    
                # select prompts
                if self.training and self.task_id_bootstrap:
                    i = int(self.e_p_length/2)
                    Ek = P_[:,:i,:].reshape((B,-1,self.embed_dim))
                    Ev = P_[:,i:,:].reshape((B,-1,self.embed_dim))
                else:
                    i = int(self.e_p_length/2)
                    Ek = P_[:,:,:i,:].reshape((B,-1,self.embed_dim))
                    Ev = P_[:,:,i:,:].reshape((B,-1,self.embed_dim))
            
            # g prompts
            g_valid = False
            if l in self.g_layers:
                g_valid = True
                j = int(self.g_p_length/2)
                p = getattr(self,f'g_p_{l}') # 0 based indexing here
                P_ = p.expand(len(x_querry),-1,-1)
                Gk = P_[:,:j,:]
                Gv = P_[:,j:,:]

            # combine prompts for prefix tuning
            if e_valid and g_valid:
                Pk = torch.cat((Ek, Gk), dim=1)
                Pv = torch.cat((Ev, Gv), dim=1)
            elif e_valid:
                Pk, Pv = Ek, Ev
            elif g_valid:
                Pk, Pv = Gk, Gv
            else:
                Pk, Pv = None, None
            
            prompt_keys.append(Pk)
            prompt_values.append(Pv)
            total_loss += loss

        return (prompt_keys, prompt_values), total_loss / len(self.e_layers)


def tensor_prompt(a, b, c=None):
    if c is None:
        p = torch.nn.Parameter(torch.FloatTensor(a,b), requires_grad=True)
    else:
        p = torch.nn.Parameter(torch.FloatTensor(a,b,c), requires_grad=True)
    nn.init.uniform_(p, -1, 1)
    return p    
