"""Run-folder guards (no torch): refuse to mix results from different runs."""
import json
import os

import numpy as np


def check_or_write_json(path, obj, keys=None):
    """Write obj if path does not exist; otherwise require the stored copy to match
    (on `keys`, or on every key when keys is None). obj must be JSON-serialisable."""
    new = json.loads(json.dumps(obj))
    if os.path.exists(path):
        with open(path) as f:
            old = json.load(f)
        for k in (keys or new):
            if old.get(k) != new.get(k):
                raise SystemExit(
                    f"STOP: {os.path.basename(path)} in this results folder differs from the current run "
                    f"(key {k!r}). Use a new --out folder, or restore the original settings.")
        return False
    with open(path, "w") as f:
        json.dump(new, f, indent=1)
    return True


def check_or_write_array(path, arr):
    """Write arr if path does not exist; otherwise require the stored array to be identical."""
    if os.path.exists(path):
        if not np.array_equal(np.load(path), arr):
            raise SystemExit(
                f"STOP: {os.path.basename(path)} in this results folder differs from the current data. "
                f"Predictions already saved here would not match these labels. Use a new --out folder.")
        return False
    np.save(path, arr)
    return True


def validate_cache_ids(cached_ids, current_ids):
    """The signal cache must have been built from exactly these records, in this order.
    Caches written before ids were stored (cached_ids is None) are accepted with a warning."""
    if cached_ids is None:
        print("WARNING: signal cache has no record ids; cannot verify it. Delete it to rebuild with ids.")
        return
    if not np.array_equal(np.asarray(cached_ids), np.asarray(current_ids)):
        raise SystemExit("STOP: signal cache does not match ptbxl_database.csv (record ids or order differ). "
                         "Delete the cache folder and run again.")
