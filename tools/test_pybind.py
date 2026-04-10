import sys
sys.path.insert(0, r"z:\git\L2\L2MapEngine\build\python\Release")
import l2map_py
import numpy as np

nodes_old    = np.loadtxt(r"\\UNRAID\Promotion\Clean_L2\Nodes_SENT_old.txt",    dtype=np.float64, delimiter=",")
nodes_new    = np.loadtxt(r"\\UNRAID\Promotion\Clean_L2\Nodes_SENT.txt",         dtype=np.float64, delimiter=",")
field_data   = np.loadtxt(r"\\UNRAID\Promotion\Clean_L2\Data_old\var_42.txt",    dtype=np.float64)
elements_old = np.loadtxt(r"\\UNRAID\Promotion\Clean_L2\Elements_SENT_old.txt",  dtype=np.float64, delimiter=",")
elements_new = np.loadtxt(r"\\UNRAID\Promotion\Clean_L2\Elements_SENT.txt",      dtype=np.float64, delimiter=",")
print(f"Nodes old shape: {nodes_old.shape}")
print(f"Nodes new shape: {nodes_new.shape}")
print(f"Elements old shape: {elements_old.shape}")
print(f"Elements new shape: {elements_new.shape}")

result = l2map_py.map_integration_points(
    nodes_new,
    elements_new,
    nodes_old,
    elements_old,
    field_data,
    element_type="Quad8",   # default
    verbose=False,
    enforce_positive=False,
    n_threads=-1,           # -1 = all cores
)

print(f"Mapped values shape: {result.values.shape}")
print(f"First few values: {result.values[:5, 0]}")

# load 
icc_raw = np.loadtxt(r"\\UNRAID\Promotion\Clean_L2\icc.txt", dtype=np.float64)
icc_raw = np.atleast_2d(icc_raw)

# Fill with NaN first so missing entries are explicit
values = np.full_like(result.values, np.nan, dtype=np.float64)
n_vals = values.shape[0]

skipped_oob = 0
skipped_bad = 0

for row in icc_raw:
    # Basic row validation
    if row.size < 3 or not np.isfinite(row[0]) or not np.isfinite(row[1]):
        skipped_bad += 1
        continue

    elem_id = int(row[0])
    point_id = int(row[1])

    # point_id in file is 1-based -> convert to 0-based
    if elem_id > 0:
        idx = (elem_id - 1) * 9 + (point_id - 1)
        mapped_value = row[2]
    else:
        if row.size <= 26:
            skipped_bad += 1
            continue
        idx = point_id - 1
        mapped_value = row[26]

    if not np.isfinite(mapped_value):
        skipped_bad += 1
        continue

    if 0 <= idx < n_vals:
        if values.ndim == 2:
            values[idx, 0] = mapped_value
        else:
            values[idx] = mapped_value
    else:
        skipped_oob += 1

if skipped_oob:
    print(f"Skipped {skipped_oob} ICC rows with out-of-range indices")
if skipped_bad:
    print(f"Skipped {skipped_bad} ICC rows with invalid data")

icc = values

# Compute metrics only where both arrays are finite
res = result.values[:, 0] if result.values.ndim == 2 else result.values
ref = icc[:, 0] if icc.ndim == 2 else icc

valid = np.isfinite(res) & np.isfinite(ref)
n_valid = int(np.count_nonzero(valid))
print(f"Valid comparison points: {n_valid}/{res.size}")

if n_valid == 0:
    print("No valid points to compare.")
else:
    error = res[valid] - ref[valid]
    rmse = np.sqrt(np.mean(error ** 2))
    l2norm = np.linalg.norm(error)
    print(f"Root Mean Squared Error: {rmse:.6e}")
    print(f"L2 Norm of Error: {l2norm:.6e}")

# Optional: replace missing ICC entries with zero after metrics
icc_filled = np.where(np.isfinite(icc), icc, 0.0)