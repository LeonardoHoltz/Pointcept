"""Pointcept inference API server.

This server loads Pointcept models from a config file and checkpoint, and accepts
raw point cloud bytes for fast inference.

Example request pattern:
    POST /infer
    meta: JSON string metadata
    point_cloud: application/octet-stream file

The response can be returned as raw binary bytes for large outputs.
"""

import json
import logging
import os
from threading import Lock
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from pointcept.models import build_model
from pointcept.utils.config import Config


logger = logging.getLogger("pointcept_api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI(title="Pointcept Inference API", version="1.0")

MODEL_CACHE: Dict[str, Dict[str, Any]] = {}
MODEL_CACHE_LOCK = Lock()

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SUPPORTED_DTYPES = {
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64,
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
    "uint8": np.uint8,
}


class InputField(BaseModel):
    key: str = Field(..., description="The model input key, e.g. coord, feat, input")
    columns: Optional[Sequence[int]] = Field(
        None,
        description="The column indices from the point cloud tensor to select for this field.",
    )
    slice: Optional[Sequence[int]] = Field(
        None,
        description="The [start, end] slice on the last axis used for this field.",
    )


class InferenceMetadata(BaseModel):
    config_path: Optional[str] = Field(
        None,
        description="Local config file path for the model. If omitted, `model_key` and `weight_path` are still used.",
    )
    weight_path: str = Field(..., description="Path to the checkpoint weight file.")
    model_key: Optional[str] = Field(
        None,
        description="Optional cache key for this model entry. If omitted, config_path+weight_path are used.",
    )
    device: str = Field(DEFAULT_DEVICE, description="Device to run inference on.")
    dtype: str = Field("float32", description="Point cloud dtype for raw byte parsing.")
    shape: Optional[Sequence[int]] = Field(
        None,
        description="Shape of the raw point cloud buffer, typically [num_points, num_features].",
    )
    point_dim: Optional[int] = Field(
        None,
        description="If shape is omitted, the number of columns in the point cloud.",
    )
    input_type: str = Field(
        "dict",
        description="How to pass point cloud to the model: raw_tensor, dict, or coord_feat.",
    )
    input_fields: Optional[Sequence[InputField]] = Field(
        None,
        description="Mappings from raw point tensor columns to model input fields.",
    )
    output_keys: Optional[Sequence[str]] = Field(
        None,
        description="If the model returns a dict, select these keys for the output.",
    )
    output_format: str = Field(
        "binary",
        description="Response format: json or binary. Use binary for large outputs.",
    )
    config_overrides: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional config overrides that are merged into the loaded config.",
    )


def get_cache_key(metadata: InferenceMetadata) -> str:
    parts = [metadata.model_key or "", metadata.config_path or "", metadata.weight_path, metadata.device]
    return "::".join([p for p in parts if p])


def resolve_path(path: str) -> str:
    if path is None:
        raise ValueError("Path is required")
    expanded = os.path.expanduser(path)
    if not os.path.isabs(expanded):
        expanded = os.path.abspath(os.path.join(os.path.dirname(__file__), expanded))
    return expanded


def load_model_from_checkpoint(
    config_path: Optional[str], weight_path: str, device: torch.device, overrides: Optional[Dict[str, Any]] = None,
) -> torch.nn.Module:
    if config_path is None:
        raise RuntimeError("config_path is required to build the model.")

    config_path = resolve_path(config_path)
    weight_path = resolve_path(weight_path)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not os.path.isfile(weight_path):
        raise FileNotFoundError(f"Weight file not found: {weight_path}")

    cfg = Config.fromfile(config_path)
    if overrides:
        cfg.merge_from_dict(overrides)

    model = build_model(cfg.model)
    model.to(device)
    checkpoint = torch.load(weight_path, map_location=device)

    # Extract state dict
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise RuntimeError("Checkpoint does not contain a valid state dict.")

    model_state_keys = set(model.state_dict().keys())
    if any(k.startswith("module.") for k in state_dict) and not any(k.startswith("module.") for k in model_state_keys):
        state_dict = {k[7:]: v for k, v in state_dict.items() if k.startswith("module.")}
    elif not any(k.startswith("module.") for k in state_dict) and any(k.startswith("module.") for k in model_state_keys):
        state_dict = {"module." + k: v for k, v in state_dict.items()}

    load_result = model.load_state_dict(state_dict, strict=False)
    if load_result.missing_keys or load_result.unexpected_keys:
        logger.warning(
            "Model load mismatch: missing_keys=%s unexpected_keys=%s",
            load_result.missing_keys,
            load_result.unexpected_keys,
        )

    model.eval()
    return model


def get_model(metadata: InferenceMetadata) -> torch.nn.Module:
    cache_key = get_cache_key(metadata)
    with MODEL_CACHE_LOCK:
        entry = MODEL_CACHE.get(cache_key)
        if entry is not None:
            return entry["model"]

    device = torch.device(metadata.device)
    model = load_model_from_checkpoint(metadata.config_path, metadata.weight_path, device, metadata.config_overrides)
    with MODEL_CACHE_LOCK:
        MODEL_CACHE[cache_key] = {"model": model, "device": device}
    return model


def parse_point_cloud_buffer(data: bytes, dtype: str, shape: Optional[Sequence[int]], point_dim: Optional[int]) -> np.ndarray:
    if dtype not in SUPPORTED_DTYPES:
        raise ValueError(f"Unsupported dtype '{dtype}'. Supported: {sorted(SUPPORTED_DTYPES)}")
    array = np.frombuffer(data, dtype=SUPPORTED_DTYPES[dtype])
    if shape is None:
        if point_dim is None:
            raise ValueError("Either shape or point_dim must be provided for raw point cloud data.")
        if array.size % point_dim != 0:
            raise ValueError("Byte length is not compatible with point_dim for the provided dtype.")
        array = array.reshape(-1, point_dim)
    else:
        shape = tuple(int(x) for x in shape)
        if np.prod(shape) != array.size:
            raise ValueError("Byte length does not match the provided shape.")
        array = array.reshape(shape)
    return array


def build_input_payload(
    points: torch.Tensor,
    metadata: InferenceMetadata,
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    if metadata.input_type == "raw_tensor":
        return points
    if metadata.input_type == "coord_feat":
        if metadata.input_fields is None or len(metadata.input_fields) == 0:
            raise ValueError("input_fields must be provided when input_type is coord_feat.")
        payload: Dict[str, torch.Tensor] = {}
        for field in metadata.input_fields:
            tensor = select_columns(points, field)
            payload[field.key] = tensor
        return payload
    if metadata.input_type == "dict":
        payload: Dict[str, torch.Tensor] = {}
        if metadata.input_fields is None or len(metadata.input_fields) == 0:
            payload["input"] = points
            return payload
        for field in metadata.input_fields:
            payload[field.key] = select_columns(points, field)
        return payload
    raise ValueError(f"Unsupported input_type: {metadata.input_type}")


def select_columns(points: torch.Tensor, field: InputField) -> torch.Tensor:
    if field.columns is not None:
        return points[:, list(field.columns)]
    if field.slice is not None:
        if len(field.slice) != 2:
            raise ValueError("Field slice must be [start, end].")
        start, end = int(field.slice[0]), int(field.slice[1])
        return points[:, start:end]
    return points


def extract_output(model_output: Any, output_keys: Optional[Sequence[str]]) -> Any:
    if output_keys is not None:
        if isinstance(model_output, dict):
            return {k: model_output[k] for k in output_keys if k in model_output}
        raise ValueError("output_keys can only be used when the model returns a dict.")
    if isinstance(model_output, dict):
        return model_output
    if isinstance(model_output, (list, tuple)):
        return model_output[0]
    return model_output


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "device": DEFAULT_DEVICE}


@app.post("/infer")
async def infer(
    meta: str = Form(...),
    point_cloud: UploadFile = File(...),
) -> StreamingResponse:
    try:
        metadata = InferenceMetadata.parse_raw(meta)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid metadata JSON: {exc}")

    try:
        raw_bytes = await point_cloud.read()
        points_np = parse_point_cloud_buffer(raw_bytes, metadata.dtype, metadata.shape, metadata.point_dim)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse point cloud bytes: {exc}")

    try:
        device = torch.device(metadata.device)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid device: {exc}")

    try:
        points = torch.from_numpy(points_np).to(device)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Cannot convert point cloud to tensor: {exc}")

    try:
        model = get_model(metadata)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {exc}")

    try:
        payload = build_input_payload(points, metadata)
        with torch.no_grad():
            model_output = model(payload)
        output = extract_output(model_output, metadata.output_keys)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {exc}")

    if isinstance(output, torch.Tensor):
        output = output.detach().cpu().numpy()
    elif isinstance(output, dict):
        output = {k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in output.items()}
    elif isinstance(output, (list, tuple)):
        output = [v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v for v in output]

    if metadata.output_format == "json":
        try:
            return JSONResponse(content={"result": output})
        except TypeError as exc:
            raise HTTPException(status_code=500, detail=f"JSON serialization failed: {exc}")

    if metadata.output_format == "binary":
        if isinstance(output, np.ndarray):
            data_bytes = output.tobytes()
            headers = {
                "X-Output-Shape": json.dumps(list(output.shape)),
                "X-Output-Dtype": str(output.dtype),
            }
            return StreamingResponse(
                content=iter([data_bytes]),
                media_type="application/octet-stream",
                headers=headers,
            )
        raise HTTPException(status_code=500, detail="Binary output only supports a single numpy array.")

    raise HTTPException(status_code=400, detail=f"Unsupported output_format: {metadata.output_format}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, log_level="info")
