#!/usr/bin/env python3
"""Remove silence from video files using FFmpeg and pydub."""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from pydub.utils import db_to_float


_SNAP_WINDOW_MS = 10
_SNAP_STEP_MS = 5
_SNAP_SEARCH_MS = 300
_PEAK_FRACTION = 0.25
_MIN_SEGMENT_MS = 33


def _snap_segment_start(
    audio: AudioSegment, start_ms: int, end_ms: int, silence_thresh: int
) -> int:
    """Walk forward from start_ms with a tight RMS window to find true speech onset.

    Pydub's detect_nonsilent considers anything above silence_thresh non-silent,
    which catches breath, lip clicks, and ambient noise as part of a "speech"
    segment — leaving pre-speech sound at the head of each clip.

    Scan the first 300ms with a 10ms window, find the peak RMS (real speech),
    and snap to the first window that exceeds _PEAK_FRACTION of peak. This
    adapts per-segment: loud speech raises the bar, soft speech lowers it.
    Floor the threshold at silence_thresh so we never return a position inside
    true silence.
    """
    search_end = min(start_ms + _SNAP_SEARCH_MS, end_ms)
    silence_floor = (
        db_to_float(silence_thresh) * audio.max_possible_amplitude
    )

    rms_values: list[tuple[int, float]] = []
    peak_rms = 0.0
    pos = start_ms
    while pos + _SNAP_WINDOW_MS <= search_end:
        rms = audio[pos : pos + _SNAP_WINDOW_MS].rms
        rms_values.append((pos, rms))
        if rms > peak_rms:
            peak_rms = rms
        pos += _SNAP_STEP_MS

    if not rms_values:
        return start_ms

    onset_thresh = max(peak_rms * _PEAK_FRACTION, silence_floor)
    for window_pos, rms in rms_values:
        if rms >= onset_thresh:
            return window_pos

    return start_ms


def extract_audio(video_path: str, audio_path: str) -> None:
    """Extract audio from video file as WAV."""
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
         "-ar", "16000", "-ac", "1", audio_path],
        check=True, capture_output=True,
    )


def detect_speaking_segments(
    audio_path: str,
    silence_thresh: int,
    min_silence_len: int = 250,
    start_padding: int = 0,
    end_padding: int = 0,
) -> list[tuple[int, int]]:
    """Detect non-silent segments in audio. Returns list of (start_ms, end_ms)."""
    audio = AudioSegment.from_wav(audio_path)
    segments = detect_nonsilent(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
    )

    if not segments:
        print("Warning: No speech detected. Check your threshold value.")
        return []

    duration_ms = len(audio)
    padded = []
    for start, end in segments:
        snapped = _snap_segment_start(audio, start, end, silence_thresh)
        if end - snapped < _MIN_SEGMENT_MS:
            continue
        padded.append((
            max(0, snapped - start_padding),
            min(duration_ms, end + end_padding),
        ))

    if not padded:
        print("Warning: All segments collapsed below frame threshold after onset snap.")
        return []

    # Merge overlapping segments
    merged = [padded[0]]
    for start, end in padded[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    return merged


def _get_video_encoder() -> list[str]:
    """Pick the best available H.264 encoder for this platform."""
    import platform
    if platform.system() == "Darwin":
        return ["-c:v", "h264_videotoolbox", "-q:v", "65"]
    try:
        result = subprocess.run(
            ["ffmpeg", "-encoders"], capture_output=True, text=True
        )
        if "h264_nvenc" in result.stdout:
            return ["-c:v", "h264_nvenc", "-cq", "23"]
    except Exception:
        pass
    return ["-c:v", "libx264", "-preset", "fast", "-crf", "18"]


def _normalize_video(
    video_path: str,
    normalized_path: str,
    key_frame_times: list[float] | None = None,
) -> None:
    """Re-encode video to H.264/AAC MP4 so segment cuts work cleanly."""
    encoder = _get_video_encoder()
    force_kf: list[str] = []
    if key_frame_times:
        times_str = ",".join(f"{t:.3f}" for t in sorted(set(key_frame_times)))
        force_kf = ["-force_key_frames", times_str]
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path,
         "-map", "0:v:0", "-map", "0:a:0",
         *encoder,
         *force_kf,
         "-c:a", "aac", "-b:a", "192k",
         "-movflags", "+faststart",
         normalized_path],
        check=True, capture_output=True,
    )


def build_trimmed_video(
    video_path: str, output_path: str, segments: list[tuple[int, int]],
    keep_segments_dir: str | None = None,
) -> list[str]:
    """Build trimmed video by normalizing, cutting segments with stream copy, and concatenating.

    Normalizing once up front lets every cut be a stream copy, which preserves
    exact A/V sync and avoids AAC priming delay on individual clip exports.

    If keep_segments_dir is provided, numbered segment files (001.mp4, 002.mp4, ...)
    are saved there for individual download. Returns list of saved segment paths.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Step 1: Normalize to H.264/AAC with keyframes forced at every segment
        # boundary so stream-copy cuts land frame-accurately.
        normalized = str(Path(tmpdir) / "normalized.mp4")
        key_frame_times = [t for start_ms, end_ms in segments
                           for t in (start_ms / 1000, end_ms / 1000)]
        _normalize_video(video_path, normalized, key_frame_times=key_frame_times)

        # Step 2: Cut each segment with stream copy (preserves exact A/V sync)
        seg_files = []
        for i, (start_ms, end_ms) in enumerate(segments):
            seg_file = str(Path(tmpdir) / f"seg_{i:04d}.mp4")
            start_s = start_ms / 1000
            duration_s = (end_ms - start_ms) / 1000
            subprocess.run(
                ["ffmpeg", "-y",
                 "-ss", f"{start_s:.3f}",
                 "-i", normalized,
                 "-t", f"{duration_s:.3f}",
                 "-c", "copy",
                 "-avoid_negative_ts", "make_zero",
                 seg_file],
                check=True, capture_output=True,
            )
            seg_files.append(seg_file)

        # Step 3: Save numbered segments if requested
        saved_segments: list[str] = []
        if keep_segments_dir:
            Path(keep_segments_dir).mkdir(parents=True, exist_ok=True)
            for i, seg_file in enumerate(seg_files):
                dest = str(Path(keep_segments_dir) / f"{i + 1:03d}.mp4")
                shutil.copy2(seg_file, dest)
                saved_segments.append(dest)

        # Step 4: Concatenate segments with concat demuxer (no re-encode)
        concat_file = str(Path(tmpdir) / "concat.txt")
        with open(concat_file, "w") as f:
            for seg_file in seg_files:
                f.write(f"file '{seg_file}'\n")

        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
             "-i", concat_file,
             "-c", "copy",
             "-movflags", "+faststart",
             output_path],
            check=True, capture_output=True,
        )

    return saved_segments


def main():
    parser = argparse.ArgumentParser(
        description="Remove silence from video files."
    )
    parser.add_argument("input", help="Input video file path")
    parser.add_argument("output", help="Output video file path")
    parser.add_argument(
        "--threshold", type=int, default=-40,
        help="Silence threshold in dB (default: -40)"
    )
    parser.add_argument(
        "--start-padding", type=int, default=0,
        help="Padding in ms kept before each speech segment (default: 0)"
    )
    parser.add_argument(
        "--end-padding", type=int, default=0,
        help="Padding in ms kept after each speech segment (default: 0)"
    )
    parser.add_argument(
        "--min-silence", type=int, default=250,
        help="Minimum silence duration in ms to detect (default: 250)"
    )
    args = parser.parse_args()

    if not Path(args.input).is_file():
        print(f"Error: Input file '{args.input}' not found.")
        sys.exit(1)

    print(f"Processing: {args.input}")
    print(
        f"Threshold: {args.threshold}dB | "
        f"Start padding: {args.start_padding}ms | "
        f"End padding: {args.end_padding}ms | "
        f"Min silence: {args.min_silence}ms"
    )
    # Step 1: Extract audio
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio_path = tmp.name

    print("Extracting audio...")
    extract_audio(args.input, audio_path)

    # Step 2: Detect speech segments
    print("Detecting speech segments...")
    segments = detect_speaking_segments(
        audio_path,
        args.threshold,
        args.min_silence,
        args.start_padding,
        args.end_padding,
    )

    Path(audio_path).unlink(missing_ok=True)

    if not segments:
        print("No segments found. Exiting.")
        sys.exit(1)

    total_kept = sum(end - start for start, end in segments) / 1000
    print(f"Found {len(segments)} speech segments ({total_kept:.1f}s total)")

    # Step 3: Build trimmed video
    print("Building trimmed video...")
    build_trimmed_video(args.input, args.output, segments)

    input_size = Path(args.input).stat().st_size / (1024 * 1024)
    output_size = Path(args.output).stat().st_size / (1024 * 1024)
    print(f"Done! {input_size:.1f}MB -> {output_size:.1f}MB")
    print(f"Output saved to: {args.output}")


if __name__ == "__main__":
    main()
