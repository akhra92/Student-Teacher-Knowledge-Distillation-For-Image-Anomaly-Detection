"""Back-compat wrapper. The implementation lives in stad.cli.score.

Prefer the console command (after `pip install -e .`) or `python -m stad.cli.score`.
"""

from stad.cli.score import main

if __name__ == "__main__":
    main()
