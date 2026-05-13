# Clippy — CLAUDE.md

Project context for Claude Code. This file helps Claude understand the codebase when assisting with development.

## Stack

- **Frontend:** Next.js (React) — `web/`
- **API:** FastAPI + Uvicorn — `api/`
- **Processing:** FFmpeg via pydub (silence detection) and subprocess (trimming/encoding)
- **CLI:** `silence_remover.py` (standalone, no server needed)

## Running Locally

```bash
# Install deps
pip install -r api/requirements.txt
cd web && npm install && cd ..

# Start both servers (production build, instant page loads)
./start.sh

# Or dev mode with hot-reload (slow first load)
./start.sh dev
```

`start.sh` builds the frontend for production on first run, then serves it via `next start`. Fixed ports: API on 8000, frontend on 3001. Logs go to `.logs/`.

## Architecture Notes

- CORS in `api/main.py` allows `localhost` and `127.0.0.1` on ports 3000 and 3001
- `next.config.ts` has `allowedDevOrigins: ["127.0.0.1"]` — required for hydration when accessing via 127.0.0.1
- Video processing is async — frontend polls `/api/status/:id` every 2 seconds
- Hardware-accelerated encoding: VideoToolbox (Mac), NVENC (NVIDIA), libx264 fallback
- `start.sh` strips sensitive env vars (Notion, Supabase, Vercel tokens, etc.) from child processes via `env -u` — prevents token leakage in `ps` output
- Silence detection has three stages in `silence_remover.py`, all applied to pydub's raw output: (1) `_snap_segment_start` walks the first 300ms with 25ms windows using a rise detector (next window ≥ 2× current) to reject pre-speech breath/click; (2) `_snap_segment_end` walks forward up to 400ms recovering fricative tails and vowel decays that pydub's 250ms window mistakes for silence; (3) `_merge_close_segments` re-joins phrases pydub split on a quiet middle syllable. Padding is applied after all three.
- The FFmpeg cut path quantizes all cut times to source frame rate (CEIL via `_quantize_to_frame`) BEFORE passing to `-force_key_frames` and before each `-ss X -c copy` cut. Without this, encoders round forced KFs to the nearest video frame and `-ss` snaps backward to a previous regular KF, producing 100-400ms of leading content + mid-syllable end cuts. See `feedback_video_encoding.md` for the why.

## Known Gotchas

- If sliders/UI isn't interactive, it's a hydration failure — check that `allowedDevOrigins` includes the hostname being used in the browser
- `api/tmp/` is where processed videos are stored temporarily — gitignored
- Next.js dev mode (`next dev`) with Turbopack on Node v25 has ~2.5 min first-compile times — this is why `start.sh` defaults to production build (`next build` + `next start`)
- Next.js 16.2.1 crashes on Node v25 with `isStableBuild is not a function` — 16.2.2 fixes this
