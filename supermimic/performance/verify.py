"""Compare optimized P4 values and actor derivatives against the retained reference."""
import argparse,json,time
from pathlib import Path
import torch
from supermimic.performance.benchmark import fixture
from supermimic.core.p4 import sample_starts,imagine
from supermimic.core.fast import FastP4
from supermimic.core.imagination import freeze_parameters


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--compile',action='store_true');p.add_argument('--precision',default='fp32',choices=['fp32','tf32','bf16']);a=p.parse_args()
    torch.set_float32_matmul_precision('highest');f=fixture();started=time.perf_counter()
    try:
        starts=sample_starts(f.replay,f.train_ids,f.heads,128)
        params=tuple(f.policy.actor.parameters())
        def gradients(result):return torch.cat([x.flatten() for x in torch.autograd.grad(result['objective'],params)])
        with freeze_parameters(f.wm,f.critic,f.amp.discriminator):
            reference=imagine(f.wm,f.policy.actor,f.norm,f.critic,f.heads,f.amp,f.features,starts,teacher=f.teacher)
            grad_ref=gradients(reference)
            engine=FastP4(f.wm,f.policy.actor,f.norm,f.critic,f.heads,f.amp,f.features,f.teacher,compile=a.compile,precision=a.precision)
            torch.set_float32_matmul_precision('high' if a.precision=='tf32' else 'highest')
            actual=engine(starts);grad=gradients(actual)
        errors={k:float((reference[k]!=actual[k]).any()) if reference[k].dtype==torch.bool
                else float((reference[k].detach()-actual[k].detach()).abs().max()) for k in reference}
        relative=float((grad-grad_ref).norm()/grad_ref.norm());cosine=float(torch.nn.functional.cosine_similarity(grad,grad_ref,dim=0))
        # BF16 is experimental: quantify drift, do not silently accept it as parity.
        critic_close=bool(torch.isclose(reference['critic_inputs'].detach(),actual['critic_inputs'].detach(),rtol=1e-5,atol=2e-5).all())
        critic_relative=float((reference['critic_inputs'].detach()-actual['critic_inputs'].detach()).norm()/reference['critic_inputs'].detach().norm())
        passed=max(errors[k] for k in ('objective','rewards','continuations'))<2e-5 and critic_close and relative<1e-3 and errors['valid_trajectories']==0
        result={'precision':a.precision,'compile':a.compile,'errors':errors,'gradient_relative_l2':relative,'gradient_cosine':cosine,'critic_inputs_relative_l2':critic_relative,'critic_inputs_rtol_1e5_atol_2e5_passed':critic_close,'fp32_parity_passed':passed,'seconds':time.perf_counter()-started}
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2));print(result,flush=True)
        if a.precision=='fp32':assert passed,result
    finally:f.env.close()


if __name__=='__main__':main()
