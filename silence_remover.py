#!/usr/bin/env python3
"""Remove silence from video files using FFmpeg and pydub."""

import argparse
import math
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from pydub.utils import db_to_float


_SCAN_WINDOW_MS = 25
_SCAN_STEP_MS = 5
_ONSET_SEARCH_MS = 300
_OFFSET_SEARCH_MS = 400
_ONSET_RISE_RATIO = 2.0
_TAIL_PEAK_FRACTION = 0.10
_TAIL_FLOOR_DB_OFFSET = -5
_OFFSET_STOP_AFTER_SILENT_MS = 100
_PHRASE_MERGE_GAP_MS = 350
_PHRASE_MERGE_FLOOR_FRACTION = 0.5
_MIN_SEGMENT_MS = 33


def _scan_rms(
    audio: AudioSegment, start_ms: int, end_ms: int,
) -> list[tuple[int, float]]:
    """Return [(window_start_ms, rms_amplitude), ...] for windows of _SCAN_WINDOW_MS."""
    out: list[tuple[int, float]] = []
    pos = start_ms
    while pos + _SCAN_WINDOW_MS <= end_ms:
        out.append((pos, audio[pos : pos + _SCAN_WINDOW_MS].rms))
        pos += _SCAN_STEP_MS
    return out


def _snap_segment_start(
    audio: AudioSegment, start_ms: int, end_ms: int, silence_thresh: int
) -> int:
    """Find the true speech onset, snapping to the START of the energy climb.

    Pydub's detect_nonsilent can include 50-200ms of breath, lip clicks, or
    ambient noise before real speech — anything above silence_thresh counts as
    non-silent in its 250ms window. Walk the first 300ms with 25ms windows and
    find the FIRST clear energy jump (next window ≥ 2× current, landing above
    the silence floor) — that jump is speech beginning. Then walk BACK over the
    contiguous above-floor climb to its start, so soft voiced onsets (h/wh/f/s,
    soft m/n/l/w/r, any quiet-to-loud ramp) are kept while only the sub-floor
    breath/silence beneath them is dropped.

    The earlier version snapped to the jump's *destination* (the top of the
    biggest rise). For any word whose soft onset ran longer than the start
    padding could recover, that clipped the first word/syllable — the loud
    vowel nucleus is the biggest 2× jump, so the snap landed mid-word. When no
    jump is found in the first 300ms the segment is already mid-speech (or the
    onset is too gradual to localize); trust pydub's start rather than advance,
    which would risk clipping the word.
    """
    search_end = min(start_ms + _ONSET_SEARCH_MS, end_ms)
    silence_floor = db_to_float(silence_thresh) * audio.max_possible_amplitude
    rms_seq = _scan_rms(audio, start_ms, search_end)
    if len(rms_seq) < 2:
        return start_ms

    peak = max(r for _, r in rms_seq)
    source_floor = max(silence_floor, peak * 0.05)

    rise_idx: int | None = None
    for i in range(len(rms_seq) - 1):
        _, src_rms = rms_seq[i]
        _, dst_rms = rms_seq[i + 1]
        if dst_rms < silence_floor:
            continue
        if dst_rms / max(src_rms, source_floor) >= _ONSET_RISE_RATIO:
            rise_idx = i
            break

    if rise_idx is None:
        return start_ms

    # From the above-floor destination window, back up over the contiguous
    # above-floor climb to its start.
    j = rise_idx + 1
    while j > 0 and rms_seq[j][1] >= silence_floor:
        j -= 1
    onset_idx = j + 1 if rms_seq[j][1] < silence_floor else j
    return rms_seq[onset_idx][0]


def _snap_segment_end(
    audio: AudioSegment, start_ms: int, end_ms: int, silence_thresh: int
) -> int:
    """Walk forward from pydub's end recovering trailing speech energy.

    Fricative tails (s, f, th), vowel decays, and consonant releases routinely
    fall 5-10 dB below pydub's silence threshold while still being audible
    speech. Pydub's 250ms RMS window declares silence as soon as the average
    dips below threshold, cutting mid-syllable. Walk forward up to 400ms with
    25ms windows, keep extending while RMS stays above the lenient tail floor
    (max(10% of segment peak, silence_thresh - 5 dB)). Stop once we've seen
    100ms of below-tail silence after the last audible frame.
    """
    duration_ms = len(audio)
    extend_end = min(end_ms + _OFFSET_SEARCH_MS, duration_ms)
    tail_lenient = (
        db_to_float(silence_thresh + _TAIL_FLOOR_DB_OFFSET)
        * audio.max_possible_amplitude
    )

    seg_rms = _scan_rms(audio, start_ms, end_ms)
    if not seg_rms:
        return end_ms
    seg_peak = max(r for _, r in seg_rms)
    tail_thresh = max(seg_peak * _TAIL_PEAK_FRACTION, tail_lenient)

    last_audible = end_ms
    consecutive_silent_ms = 0
    pos = end_ms
    while pos + _SCAN_WINDOW_MS <= extend_end:
        rms = audio[pos : pos + _SCAN_WINDOW_MS].rms
        if rms >= tail_thresh:
            last_audible = pos + _SCAN_WINDOW_MS
            consecutive_silent_ms = 0
        else:
            consecutive_silent_ms += _SCAN_STEP_MS
            if consecutive_silent_ms >= _OFFSET_STOP_AFTER_SILENT_MS:
                break
        pos += _SCAN_STEP_MS
    return last_audible


def _merge_close_segments(
    segments: list[tuple[int, int]],
    audio: AudioSegment,
    silence_thresh: int,
) -> list[tuple[int, int]]:
    """Merge consecutive segments separated by a quiet but non-silent gap.

    Pydub splits a phrase whenever a quiet syllable (sustained "uhhh", soft
    connecting vowel) drops below silence_thresh for ≥ min_silence_len. The
    speaker hasn't actually stopped, so re-join segments whose gap is small
    AND whose gap audio retains some energy above the room floor.
    """
    if not segments:
        return segments
    tail_lenient = (
        db_to_float(silence_thresh + _TAIL_FLOOR_DB_OFFSET)
        * audio.max_possible_amplitude
    )
    merge_floor = tail_lenient * _PHRASE_MERGE_FLOOR_FRACTION

    merged = [segments[0]]
    for start, end in segments[1:]:
        prev_start, prev_end = merged[-1]
        gap = start - prev_end
        if gap <= 0:
            merged[-1] = (prev_start, max(prev_end, end))
            continue
        if gap > _PHRASE_MERGE_GAP_MS:
            merged.append((start, end))
            continue
        gap_rms = audio[prev_end:start].rms
        if gap_rms >= merge_floor:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    return merged


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
    min_silence_len: int = 300,
    start_padding: int = 30,
    end_padding: int = 50,
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
    snapped: list[tuple[int, int]] = []
    for start, end in segments:
        new_start = _snap_segment_start(audio, start, end, silence_thresh)
        new_end = _snap_segment_end(audio, new_start, end, silence_thresh)
        if new_end - new_start < _MIN_SEGMENT_MS:
            continue
        snapped.append((new_start, new_end))

    if not snapped:
        print("Warning: All segments collapsed below frame threshold after onset snap.")
        return []

    # Re-join phrases that pydub split on a quiet middle syllable
    snapped = _merge_close_segments(snapped, audio, silence_thresh)

    # Apply padding
    padded = [
        (max(0, s - start_padding), min(duration_ms, e + end_padding))
        for s, e in snapped
    ]

    # Merge overlapping segments after padding
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


def _probe_fps(video_path: str) -> float:
    """Return source video frame rate as float (handles "30000/1001" form)."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", video_path],
        capture_output=True, text=True, check=True,
    ).stdout
    # iPhone .MOV files can produce output like "60/1,\n\n\n" — split on
    # both newlines and commas and take the first non-empty token.
    token = next((t.strip() for t in re.split(r"[,\n]", out) if t.strip()), "")
    if "/" in token:
        num, den = token.split("/", 1)
        return float(num) / float(den)
    return float(token)


def _quantize_to_frame(time_s: float, fps: float) -> float:
    """Snap a requested time UP to the next video-frame boundary.

    H.264 keyframes can only land on frame boundaries. Encoders snap forced
    keyframes to the nearest frame (rounding up). To make our cuts land at
    exactly where keyframes are placed, we quantize the request up-front so
    requested time, encoded keyframe position, and cut seek point all match.
    """
    return math.ceil(time_s * fps) / fps


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
    fps = _probe_fps(video_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Step 1: Quantize all cut boundaries to frame positions. H.264 KFs can
        # only land on frame boundaries, so an un-quantized request gets snapped
        # by the encoder — and then `-ss X -c copy` snaps backward to the
        # previous KF, yielding 100-400ms of leading content. Quantizing up
        # front makes requested time = encoded KF = cut seek point.
        quantized = [
            (_quantize_to_frame(s_ms / 1000, fps),
             _quantize_to_frame(e_ms / 1000, fps))
            for s_ms, e_ms in segments
        ]

        # Step 2: Normalize to H.264/AAC with keyframes forced at every cut
        # boundary so stream-copy cuts land frame-accurately.
        normalized = str(Path(tmpdir) / "normalized.mp4")
        key_frame_times = [t for s, e in quantized for t in (s, e)]
        _normalize_video(video_path, normalized, key_frame_times=key_frame_times)

        # Step 3: Cut each segment with stream copy (preserves exact A/V sync)
        seg_files = []
        for i, (start_s, end_s) in enumerate(quantized):
            seg_file = str(Path(tmpdir) / f"seg_{i:04d}.mp4")
            duration_s = max(0.001, end_s - start_s)
            subprocess.run(
                ["ffmpeg", "-y",
                 "-ss", f"{start_s:.6f}",
                 "-i", normalized,
                 "-t", f"{duration_s:.6f}",
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
        "--threshold", type=int, default=-50,
        help="Silence threshold in dB (default: -50)"
    )
    parser.add_argument(
        "--start-padding", type=int, default=30,
        help="Padding in ms kept before each speech segment (default: 30)"
    )
    parser.add_argument(
        "--end-padding", type=int, default=50,
        help="Padding in ms kept after each speech segment (default: 50)"
    )
    parser.add_argument(
        "--min-silence", type=int, default=300,
        help="Minimum silence duration in ms to detect (default: 300)"
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
