"""Dump replayable CPython set-order traces (int keys and 2-tuple keys).

Each trace records every operation and its observable outcome (pop value,
iteration order, membership result). The Julia CPythonSet emulation must
replay these traces and reproduce every observable exactly.

Dumped from CPython 3.12.16 (the same interpreter that generates the
walker fixtures) so hash-table behavior matches the oracle environment.
"""
import json
import random


rng = random.Random(20261030)


def serialize_set(s):
    return [list(x) if isinstance(x, tuple) else int(x) for x in s]


def run_int_trace(trace_id, init, nops, value_range=4000):
    ops = []
    s = set(init)
    record = {
        "trace_id": trace_id,
        "kind": "int",
        "init": [int(v) for v in init],
        "ops": ops,
    }
    for _ in range(nops):
        choice = rng.random()
        if choice < 0.30:
            v = rng.randrange(value_range)
            s.add(v)
            ops.append({"op": "add", "arg": v})
        elif choice < 0.40:
            if s:
                v = rng.choice(sorted(s))  # choose an existing member
                s.discard(v)
                ops.append({"op": "discard", "arg": v})
            else:
                ops.append({"op": "discard", "arg": rng.randrange(value_range)})
        elif choice < 0.45:
            if s:
                v = rng.choice(sorted(s))
                s.remove(v)
                ops.append({"op": "remove", "arg": v})
            else:
                try:
                    s.remove(rng.randrange(value_range))
                    ops.append({"op": "remove", "arg": -1, "result": "ok"})
                except KeyError:
                    ops.append({"op": "remove", "arg": -1, "result": "KeyError"})
        elif choice < 0.60:
            if s:
                ops.append({"op": "pop", "result": int(s.pop())})
            else:
                try:
                    s.pop()
                    ops.append({"op": "pop", "result": "ok(empty)"})
                except KeyError:
                    ops.append({"op": "pop", "result": "KeyError"})
        elif choice < 0.70:
            ops.append({"op": "iter", "result": [int(v) for v in s]})
        elif choice < 0.76:
            other_keys = [rng.randrange(value_range) for _ in range(rng.randrange(0, 40))]
            other = set(other_keys)
            res = s | other
            ops.append({"op": "or", "other": other_keys,
                        "result": [int(v) for v in res]})
            s = res
        elif choice < 0.82:
            other_keys = [rng.randrange(value_range) for _ in range(rng.randrange(0, 40))]
            other = set(other_keys)
            res = s - other
            ops.append({"op": "sub", "other": other_keys,
                        "result": [int(v) for v in res]})
            s = res
        elif choice < 0.88:
            other_keys = [rng.randrange(value_range) for _ in range(rng.randrange(0, 40))]
            other = set(other_keys)
            res = s & other
            ops.append({"op": "and", "other": other_keys,
                        "result": [int(v) for v in res]})
            s = res
        elif choice < 0.93:
            other_keys = [rng.randrange(value_range) for _ in range(rng.randrange(0, 40))]
            other = set(other_keys)
            s |= other
            ops.append({"op": "ior", "other": other_keys, "result": [int(v) for v in s]})
        elif choice < 0.97:
            other_keys = [rng.randrange(value_range) for _ in range(rng.randrange(0, 40))]
            other = set(other_keys)
            s -= other
            ops.append({"op": "isub", "other": other_keys, "result": [int(v) for v in s]})
        else:
            v = rng.randrange(value_range)
            ops.append({"op": "contains", "arg": v, "result": v in s})
    # final drain: pop until empty, recording every value
    drain = []
    while s:
        drain.append(int(s.pop()))
    ops.append({"op": "pop_drain", "result": drain})
    return record


def run_tuple_trace(trace_id, init, nops, size=6):
    """Small-value-range tuple keys: exercises tuplehash + collision clusters."""
    ops = []
    s = set(init)

    def rand_tuple():
        return (rng.randrange(size), rng.randrange(size))

    record = {
        "trace_id": trace_id,
        "kind": "tuple",
        "init": [list(v) for v in init],
        "ops": ops,
    }
    pool = [rand_tuple() for _ in range(30)]
    for _ in range(nops):
        choice = rng.random()
        if choice < 0.35:
            v = rand_tuple()
            s.add(v)
            ops.append({"op": "add", "arg": list(v)})
        elif choice < 0.45:
            if s:
                v = rng.choice(sorted(s))
                s.discard(v)
                ops.append({"op": "discard", "arg": list(v)})
            else:
                ops.append({"op": "discard", "arg": list(rand_tuple())})
        elif choice < 0.60:
            if s:
                ops.append({"op": "pop", "result": list(s.pop())})
            else:
                try:
                    s.pop()
                    ops.append({"op": "pop", "result": "ok(empty)"})
                except KeyError:
                    ops.append({"op": "pop", "result": "KeyError"})
        elif choice < 0.75:
            ops.append({"op": "iter", "result": [list(v) for v in s]})
        elif choice < 0.85:
            other_keys = rng.sample(pool, rng.randrange(0, len(pool)))
            other = set(other_keys)
            res = s | other
            ops.append({"op": "or", "other": [list(v) for v in other_keys],
                        "result": [list(v) for v in res]})
            s = res
        elif choice < 0.95:
            other_keys = rng.sample(pool, rng.randrange(0, len(pool)))
            other = set(other_keys)
            res = s & other
            ops.append({"op": "and", "other": [list(v) for v in other_keys],
                        "result": [list(v) for v in res]})
            s = res
        else:
            other_keys = rng.sample(pool, rng.randrange(0, len(pool)))
            other = set(other_keys)
            res = s - other
            ops.append({"op": "sub", "other": [list(v) for v in other_keys],
                        "result": [list(v) for v in res]})
            s = res
    drain = []
    while s:
        drain.append(list(s.pop()))
    ops.append({"op": "pop_drain", "result": drain})
    return record


def main():
    import sys
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    int_traces = []
    for i in range(40):
        n0 = rng.randrange(0, 60)
        init = [rng.randrange(200) for _ in range(n0)]  # dense -> collisions
        int_traces.append(run_int_trace(f"int{i:02d}", init, 300))
    with open(f"{out_dir}/set_traces_int.json", "w") as f:
        json.dump(int_traces, f)

    tuple_traces = []
    for i in range(20):
        n0 = rng.randrange(0, 12)
        init = [(rng.randrange(6), rng.randrange(6)) for _ in range(n0)]
        tuple_traces.append(run_tuple_trace(f"tup{i:02d}", init, 200))
    with open(f"{out_dir}/set_traces_tuple.json", "w") as f:
        json.dump(tuple_traces, f)

    # Large-set behavior: many elements, resize-heavy (used>50000 branch is
    # unreachable in the walker, but table growth past 8/32/... is not).
    big = run_int_trace("big00", [rng.randrange(100000) for _ in range(3000)], 200,
                        value_range=100000)
    with open(f"{out_dir}/set_traces_big.json", "w") as f:
        json.dump([big], f)

    print("int traces:", len(int_traces), "ops in first:",
          len(int_traces[0]["ops"]))
    print("tuple traces:", len(tuple_traces))
    print("big trace ops:", len(big["ops"]))


if __name__ == "__main__":
    main()
