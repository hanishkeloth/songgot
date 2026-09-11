"""Start a Modal function server-side and exit, so no local process holds the call.

`modal run --detach` still keeps a client attached; when macOS kills that client under memory
pressure it cancels the remote call (this cost four multi-hour GPU jobs on 2026-09-10/11).
`spawn` hands the call to Modal and returns an id immediately, so nothing local can cancel it.

    .venv/bin/python harness/spawn.py songgot-teacher-synth synth passes=120 dst=sft/synth_v6f.jsonl
"""
import ast
import sys

import modal


def main():
    app_name, fn_name, *kv = sys.argv[1:]
    kwargs = {}
    for item in kv:
        k, _, v = item.partition("=")
        try:
            kwargs[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            kwargs[k] = v
    call = modal.Function.from_name(app_name, fn_name).spawn(**kwargs)
    print(f"spawned {app_name}::{fn_name} {kwargs} -> {call.object_id}", flush=True)


if __name__ == "__main__":
    main()
