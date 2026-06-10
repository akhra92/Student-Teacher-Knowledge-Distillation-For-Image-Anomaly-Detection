"""Back-compat wrapper. The implementation lives in stad.cli.test.

Prefer the console command (after `pip install -e .`) or `python -m stad.cli.test`.
"""

from stad.cli.test import main

if __name__ == "__main__":
    main()
