"""D-245 inference-only profile guard; importing never installs a callback.

The later wrapper must pin this module and the two installed sklearn sources
before/after use. Production RF classes remain unverified: integer ndarray
[0, 1] is deliberately fail-closed. No model or artifact is read here.

Profiling is thread-specific, not an OS/native-code sandbox. A callback error
unsets profiling on that thread; rejection must terminate the attempt. The
caller alone installs/restores the returned callback with sys.setprofile.
"""
import os
from pathlib import Path
import sys


SKLEARN_SOURCE_SHA256 = {
    "calibration.py": "4c295444ea50cdd672682cc72e07110abb68c3ce81e02998cec9372449864f35",
    "preprocessing/_label.py": "a5200ac488bf8b85440783cb3ffd500d805c18e12c2dd86483985dbc5ebc29be",
}


def _lex(path):
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def _temporary_label_encoding_fit(frame, arm, sklearn_root):
    cal = sys.modules.get("sklearn.calibration")
    labels = sys.modules.get("sklearn.preprocessing._label")
    np = sys.modules.get("numpy")
    features = sys.modules.get("modules.feature_pipeline")
    if any(module is None for module in (cal, labels, np, features)):
        return False
    caller = frame.f_back
    outer = None if caller is None else caller.f_back
    origin = None if outer is None else outer.f_back
    if caller is None or outer is None or origin is None:
        return False
    encoder = frame.f_locals.get("self")
    inner = caller.f_locals.get("self")
    owner = outer.f_locals.get("self")
    y = frame.f_locals.get("y")
    return (
        arm == "rf"
        and frame.f_code is labels.LabelEncoder.fit.__code__
        and frame.f_globals is labels.__dict__
        and _lex(frame.f_code.co_filename) == _lex(sklearn_root / "preprocessing/_label.py")
        and caller.f_code is cal._CalibratedClassifier.predict_proba.__code__
        and caller.f_globals is cal.__dict__ and caller.f_lineno == 810
        and _lex(caller.f_code.co_filename) == _lex(sklearn_root / "calibration.py")
        and outer.f_code is cal.CalibratedClassifierCV.predict_proba.__code__
        and outer.f_globals is cal.__dict__ and outer.f_lineno == 530
        and origin.f_code is features.predict_positive_probability.__code__
        and origin.f_globals is features.__dict__
        and type(encoder) is labels.LabelEncoder and vars(encoder) == {}
        and cal.LabelEncoder is labels.LabelEncoder
        and type(inner) is cal._CalibratedClassifier and inner.method == "sigmoid"
        and type(owner) is cal.CalibratedClassifierCV
        and origin.f_locals.get("model") is owner
        and any(inner is member for member in owner.calibrated_classifiers_)
        and caller.f_locals.get("X") is outer.f_locals.get("X")
        and outer.f_locals.get("X") is origin.f_locals.get("features")
        and type(y) is np.ndarray and y.shape == (2,) and y.dtype.kind in "iu"
        and y is inner.classes
        and np.array_equal(y, owner.classes_)
        and np.array_equal(y, [0, 1])
    )


def make_inference_guard(code_dir, *, arm):
    """Return the canonical callback without installing it or accessing files."""
    if arm not in ("rf", "lr"):
        raise ValueError("Explicit RF or LR arm required")
    code = _lex(code_dir)
    sklearn_root = Path(sys.prefix) / "Lib/site-packages/sklearn"

    def profile(frame, event, arg):
        if event == "call":
            name = frame.f_code.co_name
            if name in ("fit", "partial_fit", "fit_transform"):
                if name != "fit" or not _temporary_label_encoding_fit(frame, arm, sklearn_root):
                    raise PermissionError("Fitting is excluded")
            filename = _lex(frame.f_code.co_filename)
            if filename == code or filename.startswith(code + os.sep):
                if (name in ("simulation_main", "_run_legacy_simulation",
                             "collect_alignment", "read_snapshot_run",
                             "run_set_c_pilot", "run_model_free_teacher",
                             "draw_figure")
                        or name.startswith("evaluate_")):
                    raise PermissionError("Excluded project entry point: " + name)

    return profile
