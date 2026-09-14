"""
Interactive inference for point clouds.

`tools/test.py` walks a dataset on disk and writes predictions out. This does
the opposite: the model is built once and kept resident, and scenes arrive one
at a time over HTTP from the viewer, which wants an answer back while a user
waits.

A scene arriving on the wire is turned into the same `data_dict` a Dataset would
hand to a DataLoader, and run through the config's own `data.test.transform` and
`test_cfg.post_transform`.

Deliberately *not* included are the two averaging steps `tools/test.py` performs:

  * `test_cfg.voxelize` in "test" mode, which splits a scene into fragments that
    each carry one point per voxel (repeating points to pad sparse cells), and
  * `test_cfg.aug_transform`, which re-runs the scene under several rotations.

`SemSegTester` sums a softmax over all of those and reports the vote. That is the
right thing for a benchmark number and the wrong thing for inspecting a model:
what comes back is a mean over a dozen forward passes, not something the network
ever produced. Here a scene is one forward pass and the reply carries the raw
logits, so what the viewer colours by is the network's actual output.

The cost of dropping voxelization is that every input point is kept, with no
per-voxel deduplication. Backbones that serialize or attend over `grid_coord`
(PTv3, and anything Point-based) are fine with that. A backbone that builds a
`spconv.SparseConvTensor` expects unique voxel indices, so if you serve one of
those and see nonsense, that is the reason.

Three request kinds are served:

  predict           per-point class probabilities for the whole scene
  ceteris_paribus   one object moved through a range of offsets, the model
                    re-run at each, reporting the object's own probabilities
  saliency          which points the model leaned on when classifying an object

Author: written for the pc-seg-interpretability viewer.
"""

import copy
from collections import OrderedDict

import numpy as np
import torch

from pointcept.datasets.transform import Compose
from pointcept.datasets.utils import collate_fn
from pointcept.models.builder import build_model
from pointcept.utils.logger import get_root_logger


# --------------------------------------------------------------------------
# turning a wire payload into the data_dict a DataLoader would produce
# --------------------------------------------------------------------------

def arrays_to_data_dict(arrays, name="scene"):
    """
    The viewer's arrays, as the assets a Dataset would have loaded off disk.

    `xyz` and `rgb` arrive as [3, N] because three contiguous per-axis arrays
    laid end to end already *are* a C-contiguous [3, N] -- nothing was
    interleaved on the way out, so nothing is de-interleaved here beyond one
    transpose to the [N, 3] the transforms expect.
    """
    if "xyz" not in arrays:
        raise ValueError("the payload carries no `xyz` array")

    xyz = np.asarray(arrays["xyz"], dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[0] != 3:
        raise ValueError(f"`xyz` must be [3, N], got {list(xyz.shape)}")
    num_points = xyz.shape[1]

    data_dict = {
        "name": name,
        "split": "test",
        "coord": np.ascontiguousarray(xyz.T, dtype=np.float32),
    }

    # Colour. Datasets carry it as 0..255 floats and let NormalizeColor scale it,
    # so hand it over in that range rather than pre-normalising here.
    if "rgb" in arrays:
        rgb = np.asarray(arrays["rgb"])
        if rgb.shape != (3, num_points):
            raise ValueError(f"`rgb` must be [3, {num_points}], got {list(rgb.shape)}")
        data_dict["color"] = np.ascontiguousarray(rgb.T, dtype=np.float32)
    else:
        # A model trained with colour still needs the channels; mid-grey is the
        # least-committal filler and matches what NormalizeColor maps to ~0.
        data_dict["color"] = np.full((num_points, 3), 127.5, dtype=np.float32)

    # Normals arrive as three separate scalar fields, one per axis, because the
    # viewer flattens every non-colour attribute to [N].
    normal = _stack_axes(arrays, "normal", num_points)
    if normal is not None:
        data_dict["normal"] = normal

    # The model does not read these at test time, but the pipeline pops them.
    data_dict["segment"] = _first_field(
        arrays, ("segment", "segment20", "label", "classification"), num_points
    )
    data_dict["instance"] = _first_field(arrays, ("instance", "object_id"), num_points)
    return data_dict


def _stack_axes(arrays, stem, num_points):
    """Rebuilds an [N, 3] asset from `<stem>_x|_y|_z` scalar fields."""
    keys = [f"{stem}_{a}" for a in ("x", "y", "z")]
    if not all(k in arrays for k in keys):
        return None
    cols = [np.asarray(arrays[k], dtype=np.float32).reshape(-1) for k in keys]
    if any(c.shape[0] != num_points for c in cols):
        return None
    return np.ascontiguousarray(np.stack(cols, axis=1), dtype=np.float32)


def _first_field(arrays, names, num_points):
    for n in names:
        if n in arrays:
            col = np.asarray(arrays[n]).reshape(-1)
            if col.shape[0] == num_points:
                return col.astype(np.int32)
    return np.full(num_points, -1, dtype=np.int32)


# --------------------------------------------------------------------------
# the engine
# --------------------------------------------------------------------------

class InferenceEngine:
    """
    A resident model plus the config's own test-time pipeline.

    Built once at startup so that a request costs a forward pass and nothing
    else -- no weight loading, no dataset scan.
    """

    def __init__(self, cfg, weight=None, device=None, grid_size=None, verbose=True):
        self.cfg = cfg
        self.logger = get_root_logger()
        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.num_classes = int(cfg.data.num_classes)

        test_cfg = cfg.data.test
        self.transform = Compose(test_cfg.get("transform", []))
        self.post_transform = Compose(test_cfg.test_cfg.get("post_transform", []))

        # `grid_coord` is normally a by-product of GridSample. Without it the
        # value is computed with GridSample's own arithmetic (see `prepare`), so
        # the lattice matches the one used in validation. The config's voxelize
        # block is read for this number only, never run.
        voxelize = test_cfg.test_cfg.get("voxelize", None)
        self.grid_size = (
            grid_size if grid_size is not None
            else (voxelize.get("grid_size") if voxelize is not None else None)
        )

        self.model = self._build_model(weight, verbose=verbose)

    # -- setup -------------------------------------------------------------

    def _build_model(self, weight, verbose=True):
        model = build_model(self.cfg.model)
        if verbose:
            n = sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.logger.info(f"Num params: {n}")
        if weight:
            checkpoint = torch.load(weight, map_location="cpu", weights_only=False)
            state = checkpoint.get("state_dict", checkpoint)
            cleaned = OrderedDict(
                # Checkpoints from a DDP run carry a "module." prefix; serving is
                # single-process, so strip it.
                (k[7:] if k.startswith("module.") else k, v) for k, v in state.items()
            )
            model.load_state_dict(cleaned, strict=True)
            if verbose:
                self.logger.info(
                    f"=> Loaded weight '{weight}' (epoch {checkpoint.get('epoch', '?')})"
                )
        elif verbose:
            self.logger.warning("=> No weight given; the model is randomly initialised.")
        return model.to(self.device).eval()

    @staticmethod
    def _mask_in_canonical(sel, canonical, meta):
        """
        Carries a scene-length boolean mask onto the canonicalised points.

        Most test pipelines only translate, so the two line up one-to-one. A
        pipeline that resamples at scene level leaves an `inverse` behind; a
        canonical point is then part of the object if any original point that
        maps to it is.
        """
        n_canonical = canonical["coord"].shape[0]
        if sel.shape[0] == n_canonical:
            return sel
        if "inverse" not in meta:
            raise ValueError(
                f"the pipeline resampled {sel.shape[0]} points to {n_canonical} "
                "without an `inverse`, so the object cannot be located"
            )
        out = np.zeros(n_canonical, dtype=bool)
        out[np.asarray(meta["inverse"])[sel]] = True
        return out

    # -- pipeline ----------------------------------------------------------

    def canonicalize(self, data_dict):
        """
        The scene-level half: the config's `transform`, which is what puts a
        scene in the frame the model was trained in.

        Kept separate from `prepare` below so that a sweep can canonicalise once
        and then move an object inside a *fixed* frame. Running it per step
        instead would let a transform like CenterShift re-centre the whole scene
        as the object moves, which is precisely the confound a ceteris paribus
        sweep exists to avoid.
        """
        data_dict = self.transform(copy.deepcopy(data_dict))
        meta = {"segment": data_dict.pop("segment"), "name": data_dict.pop("name")}
        if "origin_segment" in data_dict:
            assert "inverse" in data_dict
            meta["origin_segment"] = data_dict.pop("origin_segment")
            meta["inverse"] = data_dict.pop("inverse")
        return data_dict, meta

    def prepare(self, canonical):
        """
        The config's `post_transform`, then collation -- one input dict holding
        every point of the scene, ready for a single forward pass.
        """
        data = copy.deepcopy(canonical)
        # Without GridSample there is nothing to map back from, so the identity
        # is the honest `index`. Collect asks for it by name.
        data["index"] = np.arange(data["coord"].shape[0])
        if "grid_coord" not in data:
            if self.grid_size is None:
                raise ValueError(
                    "the pipeline needs a grid size to derive `grid_coord`; the "
                    "config has no test_cfg.voxelize.grid_size, so pass one "
                    "explicitly with --grid-size"
                )
            # Exactly GridSample's arithmetic: floor onto a lattice anchored at
            # the origin, then rebase on the lowest occupied cell. Point.sparsify()
            # has a fallback that instead anchors on the cloud's own minimum
            # corner, which is a sub-voxel phase shift of the same lattice -- on a
            # ScanNet scene it puts only ~4% of points in the cell validation used.
            # Matching GridSample keeps the grid the model sees identical to the
            # one it was validated on.
            # float64 deliberately: GridSample divides by a 0-d numpy array, which
            # promotes, whereas dividing float32 coords by a plain Python float
            # stays in float32 and lands the odd boundary point in the wrong cell.
            coord = np.asarray(data["coord"], dtype=np.float64)
            grid_coord = np.floor(coord / self.grid_size).astype(np.int64)
            data["grid_coord"] = (grid_coord - grid_coord.min(axis=0)).astype(np.int32)

        input_dict = collate_fn([self.post_transform(data)])
        for k, v in input_dict.items():
            if isinstance(v, torch.Tensor):
                input_dict[k] = v.to(self.device, non_blocking=True)
        return input_dict

    # -- predict -----------------------------------------------------------

    @torch.inference_mode()
    def predict(self, data_dict):
        """Raw per-point logits for the whole scene, [N, K]. One forward pass."""
        canonical, meta = self.canonicalize(data_dict)
        return self._predict_canonical(canonical, meta)

    @torch.inference_mode()
    def _predict_canonical(self, canonical, meta):
        """`predict` for a scene already through `canonicalize`."""
        logits = self.model(self.prepare(canonical))["seg_logits"]
        out = logits.float().cpu().numpy()
        if "inverse" in meta:
            out = out[np.asarray(meta["inverse"])]
        return out

    # -- ceteris paribus ---------------------------------------------------

    @torch.inference_mode()
    def ceteris_paribus(self, data_dict, mask, direction, offsets, on_progress=None):
        """
        Holds the scene still, slides one object along `direction`, and reports
        the object's own raw logits at each stop -- [S, K, M].

        The rest of the scene is genuinely untouched: only the masked
        coordinates move between positions, so a change in the curve can only
        come from where the object sits.
        """
        sel = np.asarray(mask).astype(bool).reshape(-1)
        if sel.sum() == 0:
            raise ValueError("the mask selects no points")
        direction = np.asarray(direction, dtype=np.float32).reshape(1, 3)
        offsets = np.asarray(offsets, dtype=np.float32).reshape(-1)

        # Canonicalise once. Every step then moves the object inside one fixed
        # frame, so the untouched points really are untouched.
        canonical, meta = self.canonicalize(data_dict)
        # The shipped test transforms only translate, so a direction vector
        # survives canonicalisation unchanged. A pipeline that rotated the scene
        # would have to rotate `direction` with it.
        moving = self._mask_in_canonical(sel, canonical, meta)
        base = np.asarray(canonical["coord"], dtype=np.float32)
        out = np.empty((offsets.size, self.num_classes, int(sel.sum())), dtype=np.float32)

        for s, offset in enumerate(offsets):
            step = dict(canonical)
            step["coord"] = base.copy()
            step["coord"][moving] += direction * float(offset)
            logits = self._predict_canonical(step, meta)
            # Masked points in ascending index order: exactly what the viewer
            # sliced out of xyz, so the two line up without a second mapping.
            out[s] = logits[sel].T
            if on_progress is not None:
                on_progress(s + 1, offsets.size)
        return out

    # -- saliency ----------------------------------------------------------

    def saliency(self, data_dict, mask, target_class=None, mode="feat"):
        """
        How much each point moved the model's opinion about one object.

        Input x gradient of the object's mean logit for `target_class`, taken
        with respect to the input features (and optionally the coordinates).
        This is Captum's `InputXGradient` in about fifteen lines; doing it
        directly avoids wrapping Pointcept's dict-in/dict-out forward in the
        tensor-in/tensor-out signature Captum wants, and keeps the sparse
        backbones working untouched.

        One forward and one backward over the whole scene, so the gradient is
        the network's own -- not a sum of gradients from a dozen views.
        """
        if mode not in ("feat", "coord", "both"):
            raise ValueError(f"unknown saliency mode {mode!r}")
        sel = np.asarray(mask).astype(bool).reshape(-1)
        if sel.sum() == 0:
            raise ValueError("the mask selects no points")

        canonical, meta = self.canonicalize(data_dict)
        in_object = torch.from_numpy(
            self._mask_in_canonical(sel, canonical, meta)
        ).to(self.device)
        input_dict = self.prepare(canonical)

        tracked = []
        for key in (("feat", "coord") if mode == "both" else (mode,)):
            if key in input_dict and input_dict[key].is_floating_point():
                input_dict[key] = input_dict[key].detach().requires_grad_(True)
                tracked.append(input_dict[key])
        if not tracked:
            raise ValueError(f"the model input has no differentiable {mode!r}")

        with torch.enable_grad():
            logits = self.model(input_dict)["seg_logits"]
            cls = (
                int(target_class)
                if target_class is not None
                else int(logits[in_object].mean(0).argmax())
            )
            score = logits[in_object, cls].mean()
            # allow_unused: a model that never reads one of these inputs yields
            # None rather than zeros, and that is a diagnosis to report at the
            # end, not a crash here.
            grads = torch.autograd.grad(
                score, tracked, retain_graph=False, allow_unused=True
            )

        contributions = [
            (g * x).abs().sum(-1) for g, x in zip(grads, tracked) if g is not None
        ]
        if contributions:
            out = torch.stack(contributions).sum(0).detach().cpu().numpy()
        else:
            out = np.zeros(canonical["coord"].shape[0], dtype=np.float32)
        out = out.astype(np.float32)
        if "inverse" in meta:
            out = out[np.asarray(meta["inverse"])]

        peak = float(np.abs(out).max())
        if peak == 0.0:
            # Silently handing back a flat field would look like "this object
            # depends on nothing"; almost always it means the score simply has no
            # differentiable path to the chosen input.
            raise ValueError(
                f"the attribution is identically zero: the model's output has no "
                f"gradient with respect to {mode!r}. Try --saliency-input "
                f"{'coord' if mode == 'feat' else 'both'}, or check that the "
                f"chosen target class is actually produced by the head."
            )
        # The viewer only colours by this, so the scale is arbitrary; normalising
        # makes one scene's ramp comparable to the next.
        return out / peak
