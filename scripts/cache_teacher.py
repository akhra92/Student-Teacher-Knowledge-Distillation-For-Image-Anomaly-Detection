"""Back-compat wrapper. The implementation lives in stad.cli.cache_teacher.

Prefer the console command (after `pip install -e .`) or `python -m stad.cli.cache_teacher`.
"""

from stad.cli.cache_teacher import main

if __name__ == "__main__":
    main()
