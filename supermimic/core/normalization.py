"""Device-resident equivalent of the native AMP running normalizer."""
import torch
from torch import nn
from rsl_rl.utils.utils import Normalizer


class TensorAMPNormalizer(nn.Module):
    def __init__(self,native,device):
        super().__init__()
        self.epsilon,self.clip_obs=native.epsilon,native.clip_obs
        for name,value in (('mean64',native.mean),('var64',native.var),('count64',native.count)):
            self.register_buffer(name,torch.as_tensor(value,device=device,dtype=torch.float64).clone())
        self.register_buffer('mean32',self.mean64.float())
        self.register_buffer('std32',(self.var64+self.epsilon).float().sqrt())

    def normalize_torch(self,x,device=None):
        return ((x-self.mean32)/self.std32).clamp(-self.clip_obs,self.clip_obs)

    @torch.no_grad()
    def update_tensor(self,x):
        # Native numpy reductions are float32 for these normalized float32 inputs.
        variance,mean=torch.var_mean(x.float(),dim=0,unbiased=False)
        mean,variance=mean.double(),variance.double()
        count=x.shape[0];total=self.count64+count;delta=mean-self.mean64
        combined=self.var64*self.count64+variance*count+delta.square()*self.count64*count/total
        self.mean64.add_(delta*count/total);self.var64.copy_(combined/total);self.count64.copy_(total)
        self.mean32.copy_(self.mean64);self.std32.copy_((self.var64+self.epsilon).float().sqrt())

    def native(self):
        result=Normalizer(self.mean64.numel(),epsilon=self.epsilon,clip_obs=self.clip_obs)
        result.mean=self.mean64.detach().cpu().numpy().copy();result.var=self.var64.detach().cpu().numpy().copy()
        result.count=float(self.count64)
        return result
