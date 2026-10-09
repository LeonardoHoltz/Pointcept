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
import os
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

import pointcept.models  # noqa: F401 -- importing registers every backbone
import torch

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


class ModelCatalogue:
    """
    The models this service is allowed to load, read from a JSON file.

    A model's **class names travel with it**. The width of the head is intrinsic
    -- a 20-class model answers with 20 logits whatever anyone configures -- but
    what index 4 *means* is not in the checkpoint, and two label spaces of the
    same width disagree about it. So each entry names its own, and switching
    model switches the legend with it rather than leaving the old names over new
    numbers.

    `config` and `weight` are relative to the pointcept checkout (the service's
    working directory); `classNames` to the repo root above it.
    """

    def __init__(self, path, repo_root=None):
        self.path = os.path.abspath(path)
        self.repo_root = repo_root or os.path.dirname(os.getcwd())
        self._read()                      # fail at startup if the file is wrong

    def _read(self):
        """
        Re-read on every use.

        The file is a few hundred bytes and the service is long-lived: adding a
        checkpoint or fixing a wrong config should not cost a restart and a
        minute of reloading the current model. Same reasoning as the viewer's
        classes.json, which is also read per request.
        """
        with open(self.path) as fh:
            doc = json.load(fh)
        entries = doc.get("models") or {}
        if not entries:
            raise ValueError(f"{self.path} lists no models")
        default = doc.get("default") or next(iter(entries))
        if default not in entries:
            raise ValueError(f"default model {default!r} is not in {self.path}")
        return entries, default

    @property
    def entries(self):
        return self._read()[0]

    @property
    def default(self):
        return self._read()[1]

    def __contains__(self, model_id):
        return model_id in self.entries

    def resolve(self, model_id):
        """A model's entry, with its paths made absolute and checked."""
        entries = self.entries
        if model_id not in entries:
            raise ValueError(
                f"unknown model {model_id!r}; this service offers "
                f"{', '.join(sorted(entries))}")
        spec = dict(entries[model_id])
        spec["id"] = model_id
        spec["config"] = os.path.abspath(spec["config"])
        if not os.path.exists(spec["config"]):
            raise ValueError(f"model {model_id!r}: no config at {spec['config']}")
        for key, base in (("weight", os.getcwd()), ("classNames", self.repo_root)):
            value = spec.get(key)
            if not value:
                continue
            resolved = value if os.path.isabs(value) else os.path.join(base, value)
            if not os.path.exists(resolved):
                raise ValueError(f"model {model_id!r}: no {key} at {resolved}")
            spec[key] = resolved
        return spec

    def listing(self, active=None):
        entries = self.entries
        return [
            {
                "id": key,
                "name": spec.get("name", key),
                "description": spec.get("description"),
                "weighted": bool(spec.get("weight")),
                "active": key == active,
            }
            for key, spec in entries.items()
        ]


class InferenceService:
    """Turns a decoded request into a reply body. Transport-agnostic."""

    def __init__(self, engine, class_names=None, saliency_mode="feat",
                 catalogue=None, model_id=None, device=None, grid_size=None):
        self.engine = engine
        self.class_names = class_names
        self.saliency_mode = saliency_mode
        self.classes = list(range(engine.num_classes))
        self.catalogue = catalogue
        self.model_id = model_id
        self._device = device
        self._grid_size = grid_size
        # One model, one CUDA context: serialise the actual work even though the
        # server accepts connections concurrently.
        self.lock = threading.Lock()

    # -- models ----------------------------------------------------------

    def describe(self):
        return {
            "service": "pointcept-inference",
            "model": self.model_id,
            "num_classes": self.engine.num_classes,
            "device": str(self.engine.device),
            "class_names": self.class_names,
        }

    def load_model(self, model_id, log=print):
        """
        Swaps the served model.

        The new one is built *before* the old is let go, so a bad config or a
        missing checkpoint leaves the service exactly as it was rather than
        taking it down -- the error goes back to the caller and the previous
        model keeps answering. Only once the new engine exists does the old one
        get dropped and its memory returned.
        """
        if self.catalogue is None:
            raise ValueError("this service was started without a model catalogue")
        spec = self.catalogue.resolve(model_id)

        with self.lock:
            if model_id == self.model_id:
                return self.describe()

            log(f"  loading {model_id} ({spec.get('name', model_id)})")
            cfg = Config.fromfile(spec["config"])
            fresh = InferenceEngine(cfg, weight=spec.get("weight"),
                                    device=self._device, grid_size=self._grid_size)
            names = load_class_names(spec.get("classNames"), fresh.num_classes)
            if names is None:
                names = fallback_class_names(cfg, fresh.num_classes)

            previous = self.engine
            self.engine = fresh
            self.class_names = names
            self.classes = list(range(fresh.num_classes))
            self.model_id = model_id

            del previous
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            log(f"  now serving {model_id}: {fresh.num_classes} classes on {fresh.device}")
            return self.describe()

    def handle(self, header, arrays, log):
        kind = header.get("request") or "predict"
        data_dict = arrays_to_data_dict(arrays, name=str(header.get("scene", "scene")))
        with self.lock:
            if kind == "saliency":
                return self._saliency(header, arrays, data_dict, log)
            if kind == "ablation":
                return self._ablation(header, arrays, data_dict, log)
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
        method = spec.get("method", "input_x_gradient")
        log(f"  saliency: instance {spec.get('instance')}, "
            f"{int(mask.sum()):,} masked points, target class {target}, method {method}")

        values = self.engine.saliency(
            data_dict, mask, target_class=target, mode=self.saliency_mode,
            method=method, baseline=spec.get("baseline", "noise"),
            steps=int(spec.get("steps", 16)),
        )
        return encode(
            {"saliency": values},
            num_points=int(values.shape[0]),
            method=method,
        )

    # -- ablation --------------------------------------------------------

    def _ablation(self, header, arrays, data_dict, log):
        """Two heatmaps, their difference, and what the object becomes without
        the points that argued hardest for A."""
        spec = header.get("ablation", {})
        mask = self._mask(arrays, data_dict)
        class_a = int(spec["class_a"])
        class_b = int(spec["class_b"])
        remove = int(spec.get("remove", 0))
        method = spec.get("method", "deeplift")
        log(f"  ablation: instance {spec.get('instance')}, {int(mask.sum()):,} masked "
            f"points, {self._name(class_a)} vs {self._name(class_b)}, "
            f"removing {remove}, method {method}")

        # Default to attributing over *both* inputs, not the server's saliency
        # default. Removing a point takes away its position as well as its
        # colour, so ranking it by a colour-only gradient answers a different
        # question than the ablation asks -- measured: a feat-only ranking
        # removed points no more damaging than the opposite ranking did.
        out = self.engine.ablation(
            data_dict, mask, class_a, class_b, remove=remove,
            mode=spec.get("input", "both"), method=method,
            baseline=spec.get("baseline", "noise"), steps=int(spec.get("steps", 16)),
        )

        before, after = out["before"], out["after"]
        log(f"    before: {self._name(class_a)} {before['per_class'][out['class_a']]:.3f} / "
            f"{self._name(class_b)} {before['per_class'][out['class_b']]:.3f} "
            f"-> majority {self._name(before['label'])}")
        if after:
            log(f"    after:  {self._name(class_a)} {after['per_class'][out['class_a']]:.3f} / "
                f"{self._name(class_b)} {after['per_class'][out['class_b']]:.3f} "
                f"-> majority {self._name(after['label'])}")

        return encode(
            {
                "attribution_a": out["attribution_a"],
                "attribution_b": out["attribution_b"],
                "attribution_diff": out["attribution_diff"],
                "removed": out["removed"].astype(np.int64),
            },
            num_points=int(data_dict["coord"].shape[0]),
            classes=self.classes,
            class_names=self.class_names,
            ablation={
                "class_a": out["class_a"], "class_b": out["class_b"],
                "object_points": out["object_points"], "removed": int(out["removed"].size),
                "method": out.get("method", method), "before": before, "after": after,
            },
        )

    def _name(self, value):
        if isinstance(self.class_names, dict):
            return self.class_names.get(int(value), f"class {value}")
        if isinstance(self.class_names, (list, tuple)) and 0 <= int(value) < len(self.class_names):
            return self.class_names[int(value)]
        return f"class {value}"

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

            # /models takes JSON, not a point cloud: a different kind of request
            # on a different path, so neither has to sniff the other's body.
            if self.path.rstrip("/").endswith("/models"):
                return self._switch_model(payload)

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

        def _json(self, status, doc):
            body = json.dumps(doc).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _switch_model(self, payload):
            """POST /models {"model": "<id>"} -- load it and serve it from now on."""
            try:
                wanted = json.loads(payload or b"{}").get("model")
            except json.JSONDecodeError as err:
                return self._json(400, {"error": f"malformed JSON: {err}"})
            if not wanted:
                return self._json(400, {"error": "give a model id as {\"model\": \"...\"}"})
            started = time.time()
            self._log(f"POST /models -> {wanted}")
            try:
                state = service.load_model(wanted, log=self._log)
            except Exception as err:  # noqa: BLE001 -- the viewer shows this text
                traceback.print_exc()
                # The previous model is still loaded and still answering.
                return self._json(400, {"error": f"{type(err).__name__}: {err}",
                                        "model": service.model_id})
            self._log(f"  switched in {time.time() - started:.1f}s")
            return self._json(200, state)

        def do_GET(self):
            # /models is the catalogue; / is a liveness probe, so the endpoint
            # can be checked from a browser.
            if self.path.rstrip("/").endswith("/models"):
                if service.catalogue is None:
                    return self._json(200, {"models": [], "active": service.model_id,
                                            "error": "no catalogue configured"})
                return self._json(200, {
                    "active": service.model_id,
                    "models": service.catalogue.listing(service.model_id),
                    **service.describe(),
                })
            return self._json(200, service.describe())

        def _log(self, message):
            if not quiet:
                print(message, flush=True)

        def log_message(self, *args):
            pass

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--config-file", default=None,
                    help="Pointcept config, as for tools/test.py. Optional when --models "
                         "supplies one")
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
    ap.add_argument("--models", default=None,
                    help="JSON catalogue of loadable models; enables GET/POST /models. "
                         "With --model, picks which of them to start on")
    ap.add_argument("--model", default=None,
                    help="id from --models to load at startup (default: the file's own)")
    args = ap.parse_args()

    # A catalogue can supply the config, weight and class names, so --config-file
    # and friends become the override rather than the only way in.
    catalogue = ModelCatalogue(args.models) if args.models else None
    model_id = None
    if catalogue is not None and (args.model or not args.config_file):
        model_id = args.model or catalogue.default
        spec = catalogue.resolve(model_id)
        args.config_file = spec["config"]
        args.weight = args.weight or spec.get("weight")
        args.class_names = args.class_names or spec.get("classNames")

    if not args.config_file:
        ap.error("give --config-file, or --models with a catalogue that supplies one")

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

    service = InferenceService(
        engine, class_names=names, saliency_mode=args.saliency_input,
        catalogue=catalogue, model_id=model_id,
        device=args.device, grid_size=args.grid_size,
    )
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
    if catalogue is not None:
        print(f"  models     {len(catalogue.entries)} in {catalogue.path}"
              + (f", serving {model_id}" if model_id else "")
              + " -- GET/POST /models", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping", flush=True)
        server.server_close()
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
