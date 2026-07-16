from pathlib import Path
import pickle
import numpy as np

base = Path(__file__).resolve().parents[4]
pkl_path = base / "resources" / "mnist.pkl"
output_dir = base / "resources" / "mnist_export"
output_dir.mkdir(parents=True, exist_ok=True)

with pkl_path.open("rb") as f:
    train_set, valid_set, test_set = pickle.load(f, encoding="latin1")

X_train, y_train = train_set
X_val, y_val = valid_set
X_test, y_test = test_set

X_train = np.asarray(X_train, dtype=np.float32) / 255.0
X_val = np.asarray(X_val, dtype=np.float32) / 255.0
X_test = np.asarray(X_test, dtype=np.float32) / 255.0

y_train = np.asarray(y_train, dtype=np.int32)
y_val = np.asarray(y_val, dtype=np.int32)
y_test = np.asarray(y_test, dtype=np.int32)

np.savetxt(output_dir / "X_train.csv", X_train, delimiter=",", fmt="%.10f")
np.savetxt(output_dir / "y_train.csv", y_train, delimiter=",", fmt="%d")
np.savetxt(output_dir / "X_val.csv", X_val, delimiter=",", fmt="%.10f")
np.savetxt(output_dir / "y_val.csv", y_val, delimiter=",", fmt="%d")
np.savetxt(output_dir / "X_test.csv", X_test, delimiter=",", fmt="%.10f")
np.savetxt(output_dir / "y_test.csv", y_test, delimiter=",", fmt="%d")

print(f"Exported {X_train.shape[0]} training rows to {output_dir}")