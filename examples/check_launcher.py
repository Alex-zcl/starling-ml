"""Два независимых процесса; outputs и seeds разделены явно."""
from tempfile import TemporaryDirectory
from pathlib import Path
from starling_ml import get_config
from starling_ml.launcher import run_experiments

if __name__=='__main__':
    with TemporaryDirectory() as directory:
        results=run_experiments([get_config('classification',max_steps=1,seed=s) for s in (10,20)],directory)
        assert [r['step'] for r in results]==[1,1]
        assert all((Path(r['directory'])/'checkpoint.pt').exists() for r in results)
        print('PASS: independent spawn workers and separate checkpoint outputs')
