"""Back-compat wrapper. The implementation lives in stad.cli.baseline_patchcore.

Prefer the console command (after `pip install -e .`) or `python -m stad.cli.baseline_patchcore`.
"""

from stad.cli.baseline_patchcore import main

if __name__ == "__main__":
    main()
