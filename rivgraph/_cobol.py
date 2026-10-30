"""COBOL walker backend: ctypes loader and march adapter.

Port contract (spec art_QEojt4gd, D2): the kernel performs numeric traversal
only. Every decision that depends on CPython set iteration order (branchpoint
clustering, emanator ordering, first-step selection) stays in Python, which
hands the kernel canonicalized inputs:

- ``skel``: padded binary skeleton as uint8 (values 0/1),
- ``bpmap``: per-pixel branchpoint verdicts (0 no, 1 yes, 255 not yet
  computed), filled lazily from :func:`rivgraph.walk.is_bp`,
- ``cant``: sorted flat indices the reference excludes while a link is
  shorter than four pixels (``cant_walk``; membership only, so the sorted
  array carries the set's full meaning).

The kernel never draws randomness and never allocates.
"""
import ctypes
from pathlib import Path

import numpy as np

# Stop codes returned by the kernel (cobol/walker.cob header comment).
ENDPOINT = 1     # raw candidates empty: link reached an endpoint
EMANATING = 2    # candidates empty after the short-link filter: delete link
BP_REACHED = 3   # single candidate was a branchpoint; path extended onto it
JUNCTION = 4     # several candidates: Python tie-breaks
BP_UNKNOWN = 5   # single candidate lacks a branchpoint verdict
OVERFLOW = 6     # path buffer would overflow; reference would spiral too

_MAX_NEIGHBORS = 8
_so_path = Path(__file__).parent / "lib" / "rivgraph_cobol.so"
_march = None


def so_path():
    """Filesystem path of the COBOL backend shared object."""
    return _so_path


def march_fn():
    """Resolve (once) the ctypes binding for the kernel entry point."""
    global _march
    if _march is None:
        path = so_path()
        if not path.exists():
            raise FileNotFoundError(
                "RIVGRAPH_COBOL=1 but the COBOL backend is not built - "
                "run `make -C cobol` (requires gnucobol3)."
            )
        lib = ctypes.CDLL(str(path))
        fn = lib.walkmarch
        fn.restype = ctypes.c_int
        fn.argtypes = [
            ctypes.POINTER(ctypes.c_ubyte),   # skel
            ctypes.POINTER(ctypes.c_ubyte),   # bpmap
            ctypes.POINTER(ctypes.c_int),     # rows
            ctypes.POINTER(ctypes.c_int),     # cols
            ctypes.POINTER(ctypes.c_int64),   # path (in/out)
            ctypes.POINTER(ctypes.c_int),     # pathlen (in/out)
            ctypes.POINTER(ctypes.c_int64),   # cant
            ctypes.POINTER(ctypes.c_int),     # ncant
            ctypes.POINTER(ctypes.c_int64),   # cap
            ctypes.POINTER(ctypes.c_int),     # nposs (out)
            ctypes.POINTER(ctypes.c_int64),   # poss (out)
            ctypes.POINTER(ctypes.c_int),     # code (out)
        ]
        _march = fn
    return _march


def march(skel, bpmap, path_buf, pathlen, cant_buf):
    """One kernel call: extend ``path_buf`` until a decision point.

    Returns ``(code, pathlen, poss)`` where ``path_buf[:pathlen]`` is the
    (possibly extended) link path and ``poss`` lists the sorted candidate
    pixels a JUNCTION or BP_UNKNOWN stop refers to.
    """
    rows, cols = skel.shape
    nposs = ctypes.c_int(0)
    poss = np.zeros(_MAX_NEIGHBORS, dtype=np.int64)
    code = ctypes.c_int(0)
    plen = ctypes.c_int(pathlen)
    rc = march_fn()(
        skel.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
        bpmap.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
        ctypes.byref(ctypes.c_int(rows)),
        ctypes.byref(ctypes.c_int(cols)),
        path_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.byref(plen),
        cant_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.byref(ctypes.c_int(cant_buf.size)),
        ctypes.byref(ctypes.c_int64(path_buf.size)),
        ctypes.byref(nposs),
        poss.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.byref(code),
    )
    if rc != 0:
        raise RuntimeError(f"COBOL walkmarch failed, rc={rc}")
    return code.value, plen.value, poss[: nposs.value].tolist()
