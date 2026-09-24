"""Независимая математика distillation, contrastive, diffusion и RL."""
import torch
import torch.nn.functional as F
from .objectives import stable,mean_objective


def distillation_objective(student,teacher,temperature=2.):
    if temperature<=0:raise ValueError("temperature must be positive")
    student,teacher=stable(student),stable(teacher).detach()
    loss=F.kl_div(F.log_softmax(student/temperature,-1),F.softmax(teacher/temperature,-1),reduction="none").sum(-1)*temperature**2
    return mean_objective(loss.sum(),loss.new_tensor(loss.numel()),("distillation",temperature))


def contrastive_objective(first,second,temperature=.1,global_negatives=False):
    if temperature<=0 or first.shape!=second.shape:raise ValueError("Invalid contrastive pair")
    first=F.normalize(stable(first),dim=-1);second=F.normalize(stable(second),dim=-1)
    candidates_first,candidates_second=first,second;offset=0
    if global_negatives:
        import torch.distributed as dist
        if dist.is_initialized():
            # Для ragged all_gather нужна отдельная padding mask; этот режим требует равных batch sizes.
            size=torch.tensor(first.shape[0],device=first.device);sizes=[torch.zeros_like(size) for _ in range(dist.get_world_size())]
            dist.all_gather(sizes,size)
            if len({int(v) for v in sizes})!=1:raise ValueError("global_negatives requires equal rank batch sizes")
            from torch.distributed.nn.functional import all_gather
            candidates_first=torch.cat(all_gather(first));candidates_second=torch.cat(all_gather(second))
            offset=dist.get_rank()*first.shape[0]
    labels=torch.arange(first.shape[0],device=first.device)+offset
    loss=(F.cross_entropy(first@candidates_second.T/temperature,labels,reduction="none")+
          F.cross_entropy(second@candidates_first.T/temperature,labels,reduction="none"))/2
    return mean_objective(loss.sum(),loss.new_tensor(loss.numel()),("contrastive",temperature,global_negatives))


def diffusion_sample(clean,steps=1000,prediction_type="epsilon"):
    """DDPM forward process; targets epsilon/sample/v задаются явно."""
    if steps<2:raise ValueError("At least two diffusion steps required")
    t=torch.randint(steps,(clean.shape[0],),device=clean.device)
    alpha=(1-torch.linspace(.0001,.02,steps,device=clean.device)).cumprod(0)[t]
    alpha=alpha.reshape(-1,*([1]*(clean.ndim-1)))
    noise=torch.randn_like(clean);noisy=alpha.sqrt()*clean+(1-alpha).sqrt()*noise
    if prediction_type=="epsilon":target=noise
    elif prediction_type=="sample":target=clean
    elif prediction_type=="v":target=alpha.sqrt()*noise-(1-alpha).sqrt()*clean
    else:raise ValueError("prediction_type must be epsilon, sample or v")
    return noisy,t,target


def generalized_advantage(rewards,values,next_values,terminated,truncated=None,gamma=.99,lam=.95):
    """Termination убирает bootstrap, truncation прерывает trace, но сохраняет bootstrap."""
    rewards,values,next_values=map(stable,(rewards,values,next_values))
    terminated=terminated.bool();truncated=torch.zeros_like(terminated) if truncated is None else truncated.bool()
    advantage=torch.zeros_like(rewards);carry=torch.zeros_like(rewards[-1])
    for t in reversed(range(len(rewards))):
        delta=rewards[t]+gamma*next_values[t]*(~terminated[t])-values[t]
        carry=delta+gamma*lam*(~(terminated[t]|truncated[t]))*carry
        advantage[t]=carry
    return advantage,advantage+values


def policy_objective(log_probability,advantage,entropy=None,entropy_weight=.01):
    loss=-log_probability*advantage.detach()
    if entropy is not None:loss=loss-entropy_weight*entropy
    return mean_objective(loss.sum(),loss.new_tensor(loss.numel()),("policy",entropy_weight))


def ppo_objective(log_probability,old_log_probability,advantage,clip=.2):
    ratio=(log_probability-old_log_probability.detach()).exp()
    loss=-torch.minimum(ratio*advantage.detach(),ratio.clamp(1-clip,1+clip)*advantage.detach())
    return mean_objective(loss.sum(),loss.new_tensor(loss.numel()),("ppo",clip))
