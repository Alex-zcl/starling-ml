# Contributing

1. Create a focused branch from `main`.
2. Install the project with `python -m pip install -e '.[detection]'`.
3. Run `python -m unittest discover -s tests -v`.
4. Update documentation and tests whenever an objective or configuration
   contract changes.
5. Open a pull request explaining the mathematical and compatibility impact.

Do not silently change loss denominators, empty-mask policies, accumulation
semantics or checkpoint guarantees. New task support belongs in pure functions
under `starling_ml.ops` and thin runtime modules, not in the Engine core.
