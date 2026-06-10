"""Back-compat wrapper. The implementation lives in stad.cli.train.

Prefer the console command (after `pip install -e .`) or `python -m stad.cli.train`.
"""

from stad.cli.train import main

if __name__ == "__main__":
    main()
