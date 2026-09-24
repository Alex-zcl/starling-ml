"""Runtime-адаптеры дополнительных задач; их алгоритмы не попадают в Engine."""
import torch
from ..core.module import Module
from ..ops.tasks import distillation_objective,contrastive_objective,diffusion_sample,policy_objective,ppo_objective,generalized_advantage


class DistillationLoss(Module):
    def setup(self,student,teacher,output="loss.distillation",temperature=2.):
        self.student,self.teacher,self.output,self.temperature=student,teacher,output,temperature
    def __call__(self):self.context[self.output]=distillation_objective(self.context[self.student],self.context[self.teacher],self.temperature).tensor()


class ContrastiveLoss(Module):
    def setup(self,first,second,output="loss.contrastive",temperature=.1,global_negatives=False):
        self.first,self.second,self.output=first,second,output;self.temperature,self.global_negatives=temperature,global_negatives
    def __call__(self):self.context[self.output]=contrastive_objective(self.context[self.first],self.context[self.second],self.temperature,self.global_negatives).tensor()


class DiffusionNoise(Module):
    def setup(self,input,noisy="batch.noisy",timesteps="batch.timesteps",target="batch.noise_target",steps=1000,prediction_type="epsilon"):
        self.input,self.noisy,self.timesteps,self.target=input,noisy,timesteps,target
        self.steps,self.prediction_type=steps,prediction_type
    def __call__(self):
        noisy,t,target=diffusion_sample(self.context[self.input],self.steps,self.prediction_type)
        self.context[self.noisy]=noisy;self.context[self.timesteps]=t;self.context[self.target]=target


class CategoricalAction(Module):
    def setup(self,logits,action="rl.action",log_probability="rl.log_probability",entropy="rl.entropy"):
        self.logits,self.action,self.log_probability,self.entropy=logits,action,log_probability,entropy
    def __call__(self):
        distribution=torch.distributions.Categorical(logits=self.context[self.logits])
        action=distribution.sample();self.context[self.action]=action
        self.context[self.log_probability]=distribution.log_prob(action);self.context[self.entropy]=distribution.entropy()


class PolicyLoss(Module):
    def setup(self,log_probability,advantage,output="train.loss",entropy=None,entropy_weight=.01):
        self.log_probability,self.advantage,self.output=log_probability,advantage,output
        self.entropy,self.entropy_weight=entropy,entropy_weight
    def __call__(self):
        self.context[self.output]=policy_objective(self.context[self.log_probability],self.context[self.advantage],
            self.context[self.entropy] if self.entropy else None,self.entropy_weight).tensor()


class PPOLoss(Module):
    def setup(self,log_probability,old_log_probability,advantage,output="loss.ppo",clip=.2):
        self.keys=(log_probability,old_log_probability,advantage);self.output,self.clip=output,clip
    def __call__(self):self.context[self.output]=ppo_objective(*(self.context[k] for k in self.keys),clip=self.clip).tensor()


class GAE(Module):
    def setup(self,rewards,values,next_values,terminated,truncated=None,advantage="rl.advantage",returns="rl.returns",gamma=.99,lam=.95):
        self.keys=(rewards,values,next_values,terminated);self.truncated=truncated
        self.advantage,self.returns,self.gamma,self.lam=advantage,returns,gamma,lam
    def __call__(self):
        advantage,returns=generalized_advantage(*(self.context[k] for k in self.keys),
            truncated=self.context[self.truncated] if self.truncated else None,gamma=self.gamma,lam=self.lam)
        self.context[self.advantage]=advantage;self.context[self.returns]=returns


class ReplayBuffer(Module):
    """Ограниченный replay с собственным сохраняемым random generator."""
    def setup(self,inputs,output="rl.replay",capacity=10000,batch_size=32,seed=42):
        self.inputs,self.output,self.capacity,self.batch_size=inputs,output,capacity,batch_size
        if capacity<1 or batch_size<1:raise ValueError("Invalid replay capacity/batch_size")
        self.items=[];self.cursor=0;self.generator=torch.Generator().manual_seed(seed)
    def __call__(self):
        item={name:self.context[key].detach().cpu().clone() for name,key in self.inputs.items()}
        if len(self.items)<self.capacity:self.items.append(item)
        else:self.items[self.cursor]=item
        self.cursor=(self.cursor+1)%self.capacity
    def reaction(self,signal,source=None,**payload):
        if signal=="sample_replay":
            if not self.items:raise ValueError("Empty replay")
            indices=torch.randint(len(self.items),(self.batch_size,),generator=self.generator).tolist()
            self.context[self.output]={k:torch.stack([self.items[i][k] for i in indices]) for k in self.inputs}
    def state_dict(self):return dict(items=self.items,cursor=self.cursor,rng=self.generator.get_state())
    def load_state_dict(self,state):self.items=state["items"];self.cursor=state["cursor"];self.generator.set_state(state["rng"])


class EnvironmentStep(Module):
    """Gymnasium-compatible boundary; replay/rollout и optimizer остаются отдельными."""
    def setup(self,environment,action=None,observation="rl.observation",reward="rl.reward",terminated="rl.terminated",truncated="rl.truncated",seed=42):
        self.environment,self.action=environment,action
        self.keys=(observation,reward,terminated,truncated);self.seed=seed;self.needs_reset=True
    def __call__(self):
        if self.needs_reset:
            observation,_=self.environment.reset(seed=self.seed);self.seed=None
            reward=0.;terminated=truncated=False
        else:
            action=self.context[self.action]
            observation,reward,terminated,truncated,_=self.environment.step(action.item() if hasattr(action,"item") else action)
        self.needs_reset=terminated or truncated
        for key,value in zip(self.keys,(observation,reward,terminated,truncated)):self.context[key]=torch.as_tensor(value)
    def close(self):self.environment.close()


class BanditReward(Module):
    def setup(self,action,target,output="rl.reward"):self.action,self.target,self.output=action,target,output
    def __call__(self):self.context[self.output]=(self.context[self.action]==self.context[self.target]).float()


class AdversarialLoss(Module):
    """Non-saturating GAN objective использует ту же чистую BCE математику."""
    def setup(self, prediction, real, output):
        self.prediction, self.real, self.output = prediction, real, output

    def __call__(self):
        from ..ops.pointwise import binary_objective
        logits = self.context[self.prediction]
        target = torch.ones_like(logits) if self.real else torch.zeros_like(logits)
        self.context[self.output] = binary_objective(logits, target).tensor()
