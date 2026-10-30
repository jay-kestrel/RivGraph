# Contributing to RivGraph

Thanks for improving RivGraph. Bug reports, documentation improvements, and
pull requests are welcome — see the README for pointers.

## Logging and output conventions

RivGraph logs through [loguru](https://loguru.readthedocs.io). To keep output
consistent and testable:

- **Log via `logger.*`, never `print()`.** Console visibility is controlled by
  the class sinks (`verbose`), not the call site.
- **Match sibling levels.** The river and delta implementations of the same
  event must log at the same level.
- **Levels**: `debug` for diagnostic detail (e.g. per-step traces), `info` for
  routine progress, `warning` for conditions the user should know about that
  do not violate a contract.
- **Contract violations go through `warnings.warn()` or exceptions**, not log
  messages. Warnings fire regardless of `verbose` and are visible to any
  embedding application; use a custom `Warning` category for conditions a
  caller may want to filter.
- When you must swallow an exception, **log it at `debug` with the original
  error** so field failures stay diagnosable.
