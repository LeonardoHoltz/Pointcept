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

    # -- attribution -------------------------------------------------------

    #: How a method forms its reference input. "noise" is a cloud of the same
    #: point count whose features are drawn to match the scene's own per-channel
    #: mean and standard deviation: the same amount of data, carrying none of
    #: the structure. "zeros" is the origin, which for normalised colour is mid
    #: grey and for coordinates is the canonical frame's centre.
    BASELINES = ("noise", "zeros")

    #: vanilla     input x gradient -- no reference at all
    #: deeplift    (input - reference) x gradient
    #: integrated  the path integral of the gradient from reference to input
    METHODS = ("vanilla", "deeplift", "integrated")

    def _reference(self, tensor, baseline, seed=0):
        """The reference input a method measures against, shaped like `tensor`."""
        if baseline == "zeros":
            return torch.zeros_like(tensor)
        if baseline != "noise":
            raise ValueError(f"unknown baseline {baseline!r}")
        # Seeded, so two runs of the same experiment give the same heatmap.
        generator = torch.Generator(device=tensor.device).manual_seed(seed)
        noise = torch.randn(
            tensor.shape, generator=generator, device=tensor.device, dtype=tensor.dtype
        )
        return noise * tensor.std(0, keepdim=True) + tensor.mean(0, keepdim=True)

    def _tracked_inputs(self, input_dict, mode):
        keys = ("feat", "coord") if mode == "both" else (mode,)
        tracked = []
        for key in keys:
            if key in input_dict and input_dict[key].is_floating_point():
                input_dict[key] = input_dict[key].detach().requires_grad_(True)
                tracked.append((key, input_dict[key]))
        if not tracked:
            raise ValueError(f"the model input has no differentiable {mode!r}")
        return tracked

    def _score(self, input_dict, in_object, target_class):
        """The object's mean logit for one class, and the class that was used."""
        logits = self.model(input_dict)["seg_logits"]
        cls = (
            int(target_class)
            if target_class is not None
            else int(logits[in_object].mean(0).argmax())
        )
        return logits[in_object, cls].mean(), cls

    def attribute(
        self, canonical, meta, in_object, target_class=None,
        mode="feat", method="deeplift", baseline="noise", steps=16, seed=0,
    ):
        """
        Signed per-point attribution for one class, in the scene's point order.

        Positive means the point pushed the object's mean logit for that class
        *up*; negative, down. The sign is the whole point here -- a difference of
        two classes' maps is what says "this point argues sofa rather than
        chair", and an absolute value would throw exactly that away.

        `method`:

        vanilla
            input x gradient. No reference: the attribution is about the input's
            own magnitude, which is why a dark point can look unimportant simply
            for being dark.

        deeplift
            (input - reference) x gradient, the Rescale rule's single-reference
            linear form. **Not** Captum's layer-wise propagation: Captum's
            DeepLift runs the input and the reference as a batch of two along
            dim 0, and for a point cloud dim 0 is *points*, not scenes -- it
            would concatenate two clouds into one and attribute the result.
            The two agree exactly where the model is locally linear and drift
            apart where it is not; `integrated` is the honest check.

        integrated
            Integrated gradients: the gradient averaged over `steps` points on
            the straight line from the reference to the input, times the
            difference. The rigorous version of the same question, at the cost
            of one forward and backward per step.
        """
        if mode not in ("feat", "coord", "both"):
            raise ValueError(f"unknown attribution input {mode!r}")
        if method not in self.METHODS:
            raise ValueError(f"unknown attribution method {method!r}")

        input_dict = self.prepare(canonical)
        tracked = self._tracked_inputs(input_dict, mode)
        references = {
            key: (None if method == "vanilla" else self._reference(x, baseline, seed))
            for key, x in tracked
        }
        inputs = {key: x for key, x in tracked}

        with torch.enable_grad():
            if method == "integrated":
                # The path integral, by the trapezoid rule over `steps` points.
                totals = {key: torch.zeros_like(x) for key, x in inputs.items()}
                cls = None
                for i in range(steps):
                    alpha = (i + 0.5) / steps
                    step_dict = dict(input_dict)
                    held = []
                    for key, x in inputs.items():
                        point = (references[key] + alpha * (x - references[key]))
                        point = point.detach().requires_grad_(True)
                        step_dict[key] = point
                        held.append((key, point))
                    score, cls = self._score(step_dict, in_object, target_class)
                    grads = torch.autograd.grad(
                        score, [t for _, t in held], retain_graph=False, allow_unused=True
                    )
                    for (key, _), g in zip(held, grads):
                        if g is not None:
                            totals[key] += g.detach() / steps
                deltas = {key: inputs[key] - references[key] for key in inputs}
                parts = [(totals[key] * deltas[key]) for key in inputs]
            else:
                score, cls = self._score(input_dict, in_object, target_class)
                grads = torch.autograd.grad(
                    score, [x for _, x in tracked], retain_graph=False, allow_unused=True
                )
                parts = []
                for (key, x), g in zip(tracked, grads):
                    if g is None:
                        continue
                    delta = x if method == "vanilla" else (x - references[key])
                    parts.append(g * delta)

        if not parts:
            raise ValueError(
                f"the model's output has no gradient with respect to {mode!r}; "
                f"try mode 'coord' or 'both'"
            )
        # Sum the feature axis, keeping the sign, then sum the inputs together.
        out = torch.stack([p.sum(-1) for p in parts]).sum(0).detach().cpu().numpy()
        out = out.astype(np.float32)
        if "inverse" in meta:
            out = out[np.asarray(meta["inverse"])]
        return out, cls

    # -- saliency ----------------------------------------------------------

    def saliency(
        self, data_dict, mask, target_class=None, mode="feat",
        method="vanilla", baseline="noise", steps=16,
    ):
        """
        How much each point moved the model's opinion about one object.

        The magnitude of `attribute` above, normalised to 0..1 -- the viewer
        colours by it, so only the shape of the field matters, and normalising
        makes one scene's ramp comparable to the next.
        """
        sel = np.asarray(mask).astype(bool).reshape(-1)
        if sel.sum() == 0:
            raise ValueError("the mask selects no points")

        canonical, meta = self.canonicalize(data_dict)
        in_object = torch.from_numpy(
            self._mask_in_canonical(sel, canonical, meta)
        ).to(self.device)

        values, _ = self.attribute(
            canonical, meta, in_object, target_class=target_class,
            mode=mode, method=method, baseline=baseline, steps=steps,
        )
        out = np.abs(values)
        peak = float(out.max())
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
        return out / peak

    # -- ablation ----------------------------------------------------------

    def ablation(
        self, data_dict, mask, class_a, class_b, remove=0,
        mode="feat", method="deeplift", baseline="noise", steps=16,
    ):
        """
        Does an object stop being a sofa if you take away the points that argue
        it is one?

        Attributes the object's mean logit for `class_a` and for `class_b`,
        subtracts the two maps, deletes the `remove` object points that argue
        hardest for A over B, and segments the scene again without them. What
        comes back is both heatmaps, their difference, which points went, and
        the object's mean probability for each class before and after -- so the
        claim "A turned into B" is a number, not an impression.

        The MNIST original erased pixels; here the points genuinely leave the
        cloud, which is the same operation for a sparse representation. The
        scene is re-canonicalised for the second pass, because removing points
        changes the voxel grid and pretending otherwise would score a cloud the
        model was never given.
        """
        sel = np.asarray(mask).astype(bool).reshape(-1)
        object_idx = np.flatnonzero(sel)
        if object_idx.size == 0:
            raise ValueError("the mask selects no points")
        remove = int(max(0, min(remove, object_idx.size)))

        canonical, meta = self.canonicalize(data_dict)
        in_object = torch.from_numpy(
            self._mask_in_canonical(sel, canonical, meta)
        ).to(self.device)

        common = dict(mode=mode, method=method, baseline=baseline, steps=steps)
        attr_a, cls_a = self.attribute(canonical, meta, in_object, target_class=class_a, **common)
        attr_b, cls_b = self.attribute(canonical, meta, in_object, target_class=class_b, **common)
        diff = attr_a - attr_b

        # Only the object's own points are candidates: removing scene points
        # would be a different experiment (and a much easier way to break a
        # prediction).
        order = object_idx[np.argsort(-diff[object_idx], kind="stable")]
        removed = order[:remove]

        before = self._object_probs(self._predict_canonical(canonical, meta), sel, (cls_a, cls_b))

        keep = np.ones(sel.shape[0], dtype=bool)
        keep[removed] = False
        after = None
        if remove > 0:
            reduced = {
                k: (v[keep] if isinstance(v, np.ndarray) and v.shape[:1] == sel.shape else v)
                for k, v in copy.deepcopy(data_dict).items()
            }
            logits_after = self.predict(reduced)
            after = self._object_probs(logits_after, sel[keep], (cls_a, cls_b))

        return {
            "attribution_a": attr_a,
            "attribution_b": attr_b,
            "attribution_diff": diff.astype(np.float32),
            "removed": removed.astype(np.int64),
            "class_a": cls_a,
            "class_b": cls_b,
            "object_points": int(object_idx.size),
            "before": before,
            "after": after,
        }

    @staticmethod
    def _object_probs(logits, sel, classes):
        """Mean softmax probability over an object's points, per class asked for."""
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        rows = probs[sel]
        mean = rows.mean(axis=0)
        return {
            "per_class": {int(c): float(mean[int(c)]) for c in classes},
            "label": int(np.bincount(rows.argmax(axis=1), minlength=mean.shape[0]).argmax()),
            "points": int(rows.shape[0]),
        }
