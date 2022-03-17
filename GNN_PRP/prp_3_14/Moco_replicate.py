import torch
from torch import nn
import torch.nn.functional as F


##############################################################################################
##############################################################################################
##                                                                                          ##
##                                        UTILS                                             ##
##                                                                                          ##
##############################################################################################
##############################################################################################

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


##############################################################################################
##############################################################################################
##                                                                                          ##
##                                       MoCo v3                                            ##
##                                                                                          ##
##############################################################################################
##############################################################################################

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output



class MoCo(nn.Module):
    def __init__(self, base_encoder, dim=128, 
                 K:int=2**16, M=0.999, T=0.07, mlp=False):
        """
        Args:
            base_encoder (_type_): encoder
            dim (int, optional): feature dimensions. Defaults to 128.
            K (int, optional): queue size. Defaults to 2**16.
            M (float, optional): MoCo momentum. Defaults to 0.999.
            T (float, optional): temperature. Defaults to 0.07.
            mlp (bool, optional): MLP. Defaults to False.
        """
        
        super(MoCo, self).__init__()
        
        self.K = K
        self.T = T
        self.M = M
        
        # encoders
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)
        
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.require_grad = False
        
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = F.normalize(self.queue, dim=0)
        
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        update the encoder_k with the momentum
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data += (1 - self.M) * (param_q.data - param_k.data)
    

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """

        Args:
            keys (Tensor): _description_
        """
        keys = concat_all_gather(keys)
        
        batch_size = keys.shape[0]
        
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0
        
        self.queue[:, ptr:ptr+batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K
        
        self.queue_ptr[0] = ptr
        
    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]
        
        num_gpus = batch_size_all // batch_size_this
        
        # random shuffle idx
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)
        
        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]
        
        return x_gather[idx_this], idx_unshuffle
    
    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]
        
        num_gpus = batch_size_all // batch_size_this
        
        # restore index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]
        
        return x_gather[idx_this]
            
    
    def forward(self, im_q, im_k):
        """_summary_

        Args:
            im_q (_type_): a batch of query images
            im_k (_type_): a batch of key images
        Output:
            logits, targets
        """
        
        q = self.encoder_q(im_q)
        q = F.normalize(q, dim=1)
        
        with torch.no_grad():
            self._momentum_update_key_encoder()
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            k = self.encoder_k(im_k)
            k = F.normalize(k, dim=1)
            
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)
        
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc, ck->nk', [q, self.queue.clone().detach()])
        
        logits = torch.cat([l_pos, l_neg], dim=1) / self.T
        
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        self._dequeue_and_enqueue(k)
        
        return logits, labels
            