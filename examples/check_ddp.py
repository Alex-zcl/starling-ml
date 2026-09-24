"""Запуск: python examples/check_ddp.py; два Gloo ranks сравниваются с full batch."""
from datetime import timedelta
from pathlib import Path
import tempfile
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from starling_ml.ops.pointwise import categorical_loss_objective
from starling_ml.ops.losses import overlap_objective


def worker(rank, rendezvous, output):
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=20))
    try:
        torch.set_num_threads(1)
        torch.manual_seed(19)
        model=torch.nn.Linear(4,3,bias=False).double()
        model=torch.nn.parallel.DistributedDataParallel(model)
        x=torch.arange(20,dtype=torch.float64).reshape(5,4)/20
        target=torch.tensor([0,1,-100,2,0])
        sl=slice(0,2) if rank==0 else slice(2,5)
        loss=categorical_loss_objective(model(x[sl]),target[sl]).value(distributed=True)
        loss.backward()
        if rank==0:torch.save(dict(loss=loss.detach(),grad=model.module.weight.grad),output)
    finally:
        dist.destroy_process_group()


def main():
    with tempfile.TemporaryDirectory() as d:
        output=str(Path(d)/'result.pt')
        mp.spawn(worker,args=(str(Path(d)/'rendezvous'),output),nprocs=2,join=True)
        distributed=torch.load(output,weights_only=True)
    torch.manual_seed(19)
    model=torch.nn.Linear(4,3,bias=False).double()
    x=torch.arange(20,dtype=torch.float64).reshape(5,4)/20
    loss=categorical_loss_objective(model(x),torch.tensor([0,1,-100,2,0])).value();loss.backward()
    torch.testing.assert_close(distributed['loss'],loss)
    torch.testing.assert_close(distributed['grad'],model.weight.grad)
    print('PASS: 2-rank Gloo loss and gradients equal full batch with unequal valid counts')


if __name__=='__main__':main()
