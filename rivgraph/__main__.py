"""Allow ``python -m rivgraph`` to invoke the command-line interface."""
from rivgraph.cli import main

if __name__ == '__main__':
    raise SystemExit(main())
