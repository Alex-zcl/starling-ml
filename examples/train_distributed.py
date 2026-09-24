"""torchrun --standalone --nproc-per-node=2 examples/train_distributed.py"""
from starling_ml import Engine, get_config
from starling_ml.distributed import distributed_session, distributed_config

if __name__ == '__main__':
    with distributed_session() as rank:
        config = distributed_config(get_config('classification', max_steps=3))
        engine = Engine(config).run()
        if rank == 0:
            print('Completed', engine.context.data['run.step'], 'DDP updates')
