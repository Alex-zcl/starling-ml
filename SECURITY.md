# Security policy

## Supported versions

Security fixes are currently provided for the latest published release.

## Reporting a vulnerability

Please do not open a public issue for a suspected vulnerability. Use GitHub's
private vulnerability reporting for `Alex-zcl/starling-ml`. Include a minimal
reproduction, affected versions and the potential impact when possible.

Training checkpoints are serialized with PyTorch and may execute code while
loading. Only load checkpoints from trusted sources. Configuration files may
contain Python import paths and must also be treated as trusted input.
