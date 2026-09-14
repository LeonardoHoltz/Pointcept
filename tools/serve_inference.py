"""
Serves a Pointcept model to the point cloud interpretability viewer.

The model is built once, at startup, and then waits. Each request carries one
scene in the viewer's binary framing; the reply carries the arrays it asked for,
in the same framing.

    python tools/serve_inference.py \
        --config-file configs/scannet/semseg-pt-v3m1-0-base.py \
        --weight exp/scannet/semseg-pt-v3m1-0-base/model/model_best.pth \
        --port 8500

Then point the viewer's Segmentation tab at http://127.0.0.1:8500/.

Requests are distinguished by the header's `request` field:

    (absent)          per-point labels, scores and probabilities
    ceteris_paribus   one object swept along a direction
    saliency          attribution for one object

Author: written for the pc-seg-interpretability viewer.
"""

import argparse
import json
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

import pointcept.models  # noqa: F401 -- importing registers every backbone
from pointcept.engines.inference import InferenceEngine, arrays_to_data_dict
from pointcept.utils.config import Config
from pointcept.utils.logger import get_root_logger
from pointcept.utils.pcit_protocol import decode, encode


# --------------------------------------------------------------------------
# class names
# --------------------------------------------------------------------------

def load_class_names(path, num_classes):
    """
    Names for the classes, as the viewer's legend wants them: {"<id>": "<name>"}.

    Accepts a plain list, a flat id->name mapping, or the viewer's own
    `classes.json` (in which case the field with the right number of classes is
    used), so the same file can describe the dataset on both sides.
    """
    if path is None:
        return None
    with open(path) as f:
        doc = json.load(f)

    if isinstance(doc, list):
        return {str(i): str(n) for i, n in enumerate(doc[:num_classes])}

    if "fields" in doc:
        best = None
        for spec in doc["fields"].values():
            classes = spec.get("classes")
            if not classes:
                continue
            # Ignore an "ignore" entry such as -1 when counting; the model only
            # ever emits 0..K-1.
            positive = {k: v for k, v in classes.items() if int(k) >= 0}
            if len(positive) == num_classes:
                best = positive
                break
            if best is None:
                best = positive
        if best is None:
            return None
        return {k: v.get("name", f"class {k}") if isinstance(v, dict) else str(v)
                for k, v in best.items()}

    return {str(k): (v.get("name") if isinstance(v, dict) else str(v))
            for k, v in doc.items()}


def fallback_class_names(cfg, num_classes):
    """ScanNet's own label lists, when the config names one of those datasets."""
    dataset = str(cfg.data.test.get("type", ""))
    try:
        from pointcept.datasets.preprocessing.scannet.meta_data.scannet200_constants import (
            CLASS_LABELS_20, CLASS_LABELS_200,
        )
    except ImportError:
        return None
    if dataset == "ScanNetDataset" and num_classes == len(CLASS_LABELS_20):
        return {str(i): n for i, n in enumerate(CLASS_LABELS_20)}
    if dataset == "ScanNet200Dataset" and num_classes == len(CLASS_LABELS_200):
        return {str(i): n for i, n in enumerate(CLASS_LABELS_200)}
    return None


# --------------------------------------------------------------------------
# request handling
# --------------------------------------------------------------------------

# Python spellings first: `json.loads("False")` raises, and falling back to the
# raw string would set a flag to the truthy "False".
_LITERALS = {"True": True, "False": False, "None": None,
             "true": True, "false": False, "null": None}


def _parse_option(value):
    """Turns a `--options key=value` right-hand side into a Python value."""
    if value in _LITERALS:
        return _LITERALS[value]
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


class InferenceService:
    """Turns a decoded request into a reply body. Transport-agnostic."""

    def __init__(self, engine, class_names=None, saliency_mode="feat"):
        self.engine = engine
        self.class_names = class_names
        self.saliency_mode = saliency_mode
        self.classes = list(range(engine.num_classes))
        # One model, one CUDA context: serialise the actual work even though the
        # server accepts connections concurrently.
        self.lock = threading.Lock()

    def handle(self, header, arrays, log):
        kind = header.get("request") or "predict"
        data_dict = arrays_to_data_dict(arrays, name=str(header.get("scene", "scene")))
        with self.lock:
            if kind == "saliency":
                return self._saliency(header, arrays, data_dict, log)
            if kind == "ceteris_paribus":
                return self._ceteris(header, arrays, data_dict, log)
            return self._predict(header, arrays, data_dict, log)

    # -- predict ---------------------------------------------------------

    def _predict(self, header, arrays, data_dict, log):
        n = data_dict["coord"].shape[0]
        log(f"  predict: {n:,} points, "
            + ", ".join(f"{k}{tuple(v.shape)}:{v.dtype}" for k, v in arrays.items()))

        logits = self.engine.predict(data_dict)                # (N, K), raw
        labels = logits.argmax(axis=1).astype(np.int32)
        # A 0..1 confidence for the viewer's ramp. The softmax is over this one
        # forward pass, so it is a view of the logits rather than an average of
        # anything -- and `logits` itself goes back untouched alongside it.
        scores = _softmax(logits).max(axis=1).astype(np.float32)

        log(f"  -> {len(np.unique(labels))} distinct classes, "
            f"logits in [{logits.min():.2f}, {logits.max():.2f}]")
        return encode(
            {"labels": labels, "scores": scores,
             # The viewer wants [C, N]; the network gives [N, C].
             "logits": np.ascontiguousarray(logits.T, dtype=np.float32)},
            num_points=int(n),
            classes=self.classes,
            class_names=self.class_names,
        )

    # -- ceteris paribus -------------------------------------------------

    def _ceteris(self, header, arrays, data_dict, log):
        spec = header["ceteris_paribus"]
        mask = self._mask(arrays, data_dict)
        offsets = spec["offsets"]
        log(f"  ceteris paribus: instance {spec.get('instance')}, "
            f"{int(mask.sum()):,} masked points, {len(offsets)} positions, one request")

        out = self.engine.ceteris_paribus(
            data_dict, mask, spec.get("direction", [0, 0, 1]), offsets,
            on_progress=lambda i, total: log(f"    position {i}/{total}"),
        )
        # `logits`, not `probs`: one forward per position, sent raw, and the
        # viewer applies the softmax itself when it plots a class.
        return encode(
            {"logits": out},
            num_points=int(header.get("num_points", data_dict["coord"].shape[0])),
            classes=self.classes,
            class_names=self.class_names,
        )

    # -- saliency --------------------------------------------------------

    def _saliency(self, header, arrays, data_dict, log):
        spec = header.get("saliency", {})
        mask = self._mask(arrays, data_dict)
        target = spec.get("target_class")
        log(f"  saliency: instance {spec.get('instance')}, "
            f"{int(mask.sum()):,} masked points, target class {target}")

        values = self.engine.saliency(
            data_dict, mask, target_class=target, mode=self.saliency_mode
        )
        return encode({"saliency": values}, num_points=int(values.shape[0]))

    # -- shared ----------------------------------------------------------

    @staticmethod
    def _mask(arrays, data_dict):
        if "mask" not in arrays:
            raise ValueError("the request carries no `mask` array")
        mask = np.asarray(arrays["mask"]).reshape(-1).astype(bool)
        n = data_dict["coord"].shape[0]
        if mask.shape[0] != n:
            raise ValueError(f"`mask` has {mask.shape[0]} entries but the scene has {n} points")
        return mask


def make_handler(service, quiet=False):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            length = int(self.headers.get("content-length", 0))
            payload = self.rfile.read(length)
            started = time.time()
            try:
                header, arrays = decode(payload)
                body = service.handle(header, arrays, self._log)
                self._log(f"  done in {time.time() - started:.1f}s")
                self.send_response(200)
                self.send_header("content-type", "application/octet-stream")
            except Exception as err:  # noqa: BLE001 -- the viewer shows this text
                traceback.print_exc()
                body = json.dumps({"error": f"{type(err).__name__}: {err}"}).encode()
                self.send_response(400)
                self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            # A liveness probe, so the endpoint can be checked from a browser.
            body = json.dumps({
                "service": "pointcept-inference",
                "num_classes": service.engine.num_classes,
                "device": str(service.engine.device),
            }).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _log(self, message):
            if not quiet:
                print(message, flush=True)

        def log_message(self, *args):
            pass

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--config-file", required=True, help="Pointcept config, as for tools/test.py")
    ap.add_argument("--weight", default=None, help="checkpoint to serve")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8500)
    ap.add_argument("--device", default=None, help="cuda, cuda:1, cpu (default: cuda if present)")
    ap.add_argument("--grid-size", type=float, default=None,
                    help="voxel size used to derive `grid_coord` (default: the config's "
                         "test_cfg.voxelize.grid_size). The scene is not voxelized -- this "
                         "only sets the grid the model indexes against")
    ap.add_argument("--class-names", default=None,
                    help="JSON with class names: a list, an id->name map, or the viewer's "
                         "own classes.json")
    ap.add_argument("--saliency-input", default="feat", choices=("feat", "coord", "both"),
                    help="what saliency is taken with respect to (default: the input features)")
    ap.add_argument("--options", nargs="+", action="append", default=None,
                    help="config overrides, e.g. --options data.num_classes=20")
    args = ap.parse_args()

    cfg = Config.fromfile(args.config_file)
    if args.options:
        flat = {}
        for group in args.options:
            for item in group:
                key, _, value = item.partition("=")
                flat[key] = _parse_option(value)
        cfg.merge_from_dict(flat)

    logger = get_root_logger()
    logger.info("=> Building model ...")
    engine = InferenceEngine(cfg, weight=args.weight, device=args.device,
                             grid_size=args.grid_size)

    names = load_class_names(args.class_names, engine.num_classes)
    if names is None:
        names = fallback_class_names(cfg, engine.num_classes)

    service = InferenceService(engine, class_names=names, saliency_mode=args.saliency_input)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(service))

    print(f"pointcept inference service on http://{args.host}:{args.port}", flush=True)
    print(f"  config     {args.config_file}", flush=True)
    print(f"  weight     {args.weight or '(none -- random weights)'}", flush=True)
    print(f"  device     {engine.device}", flush=True)
    print(f"  classes    {engine.num_classes}"
          + (f" ({', '.join(list(names.values())[:4])}, ...)" if names else " (unnamed)"),
          flush=True)
    print(f"  grid size  {engine.grid_size}", flush=True)
    print(f"  mode       one forward pass per scene, raw logits "
          f"(no fragments, no TTA)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping", flush=True)
        server.server_close()
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
