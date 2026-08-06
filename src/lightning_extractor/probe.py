from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess


class ProbeError(RuntimeError):
    pass


def probe_video(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise ProbeError(f"Video does not exist: {path}")
    executable = shutil.which("ffprobe")
    if executable is None:
        raise ProbeError("ffprobe is required but was not found in PATH")
    command = [
        executable,
        "-v", "error",
        "-show_entries",
        "format=duration,size,bit_rate,format_name:stream=index,codec_type,codec_name,width,height,avg_frame_rate,pix_fmt,color_space,sample_rate,channels",
        "-of", "json",
        str(path),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode:
        raise ProbeError(result.stderr.strip() or f"ffprobe failed for {path}")
    data = json.loads(result.stdout)
    video_stream = next(
        (stream for stream in data.get("streams", []) if stream.get("codec_type") == "video"),
        None,
    )
    if video_stream is None:
        raise ProbeError(f"No video stream found in {path}")
    rate = str(video_stream.get("avg_frame_rate", "0/1"))
    numerator, denominator = (float(part) for part in rate.split("/", 1))
    fps = numerator / denominator if denominator else 0.0
    return {
        "path": str(path.resolve()),
        "name": path.name,
        "duration_seconds": float(data.get("format", {}).get("duration", 0.0)),
        "size_bytes": int(data.get("format", {}).get("size", path.stat().st_size)),
        "bit_rate": int(data.get("format", {}).get("bit_rate", 0)),
        "format": data.get("format", {}).get("format_name"),
        "video": {**video_stream, "fps": fps},
        "has_audio": any(
            stream.get("codec_type") == "audio" for stream in data.get("streams", [])
        ),
    }

