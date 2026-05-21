"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import Dropzone from "./components/dropzone";
import Controls from "./components/controls";
import StatusBar from "./components/status-bar";
import ClippyLogo from "./components/clippy-logo";

type Status = "idle" | "uploading" | "processing" | "done" | "error";

const API = "http://localhost:8000";

export default function Home() {
  const [files, setFiles] = useState<File[]>([]);
  const [threshold, setThreshold] = useState(-50);
  const [startPadding, setStartPadding] = useState(50);
  const [endPadding, setEndPadding] = useState(50);
  const [minSilence, setMinSilence] = useState(300);

  const [status, setStatus] = useState<Status>("idle");
  const [step, setStep] = useState("");
  const [segments, setSegments] = useState<number | null>(null);
  const [inputSizeMb, setInputSizeMb] = useState<number | null>(null);
  const [outputSizeMb, setOutputSizeMb] = useState<number | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [batch, setBatch] = useState(false);
  const [totalFiles, setTotalFiles] = useState<number>(0);
  const [currentFile, setCurrentFile] = useState<number>(0);
  const [trimmedFiles, setTrimmedFiles] = useState<{ index: number; filename: string; size_mb: number }[]>([]);
  const [segmentCount, setSegmentCount] = useState<number | null>(null);

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  useEffect(() => {
    return () => stopPolling();
  }, [stopPolling]);

  const pollStatus = useCallback(
    (id: string) => {
      pollRef.current = setInterval(async () => {
        try {
          const res = await fetch(`${API}/api/status/${id}`);
          if (res.status === 404) {
            setStatus("error");
            setStep("Job not found. Please try again.");
            stopPolling();
            return;
          }
          if (!res.ok) return;
          const data = await res.json();

          setStep(data.step);
          setSegments(data.segments);

          if (data.batch) {
            setBatch(true);
            setTotalFiles(data.total_files);
            setCurrentFile(data.current_file);
          }

          if (data.status === "done") {
            setStatus("done");
            setInputSizeMb(data.input_size_mb);
            setOutputSizeMb(data.output_size_mb);
            if (data.trimmed_files) setTrimmedFiles(data.trimmed_files);
            if (data.segment_count) setSegmentCount(data.segment_count);
            stopPolling();
          } else if (data.status === "error") {
            setStatus("error");
            stopPolling();
          }
        } catch {
          // Network error — keep polling
        }
      }, 2000);
    },
    [stopPolling]
  );

  const handleProcess = useCallback(async () => {
    if (files.length === 0) return;

    setStatus("uploading");
    setStep("Uploading video" + (files.length > 1 ? "s" : "") + "...");
    setSegments(null);
    setInputSizeMb(null);
    setOutputSizeMb(null);
    setBatch(files.length > 1);
    setTotalFiles(files.length);
    setCurrentFile(0);

    try {
      const formData = new FormData();

      if (files.length === 1) {
        // Single file — use original endpoint
        formData.append("file", files[0]);
        formData.append("threshold", threshold.toString());
        formData.append("start_padding", startPadding.toString());
        formData.append("end_padding", endPadding.toString());
        formData.append("min_silence", minSilence.toString());

        const res = await fetch(`${API}/api/upload`, {
          method: "POST",
          body: formData,
        });

        if (!res.ok) {
          const err = await res.json().catch(() => ({ detail: "Upload failed" }));
          throw new Error(err.detail || "Upload failed");
        }

        const data = await res.json();
        setJobId(data.job_id);
        setStatus("processing");
        setStep("Starting processing...");
        pollStatus(data.job_id);
      } else {
        // Multiple files — use batch endpoint
        for (const file of files) {
          formData.append("files", file);
        }
        formData.append("threshold", threshold.toString());
        formData.append("start_padding", startPadding.toString());
        formData.append("end_padding", endPadding.toString());
        formData.append("min_silence", minSilence.toString());

        const res = await fetch(`${API}/api/upload-batch`, {
          method: "POST",
          body: formData,
        });

        if (!res.ok) {
          const err = await res.json().catch(() => ({ detail: "Upload failed" }));
          throw new Error(err.detail || "Upload failed");
        }

        const data = await res.json();
        setJobId(data.job_id);
        setStatus("processing");
        setStep("Starting batch processing...");
        pollStatus(data.job_id);
      }
    } catch (err) {
      setStatus("error");
      setStep(err instanceof Error ? err.message : "Upload failed");
    }
  }, [files, threshold, startPadding, endPadding, minSilence, pollStatus]);

  const handleReprocess = useCallback(() => {
    stopPolling();
    setStatus("idle");
    setStep("");
    setSegments(null);
    setInputSizeMb(null);
    setOutputSizeMb(null);
    setJobId(null);
    setBatch(false);
    setTotalFiles(0);
    setCurrentFile(0);
    setTrimmedFiles([]);
    setSegmentCount(null);
  }, [stopPolling]);

  const handleReset = useCallback(() => {
    handleReprocess();
    setFiles([]);
  }, [handleReprocess]);

  const [shuttingDown, setShuttingDown] = useState(false);

  const handleShutdown = useCallback(async () => {
    if (!window.confirm("Shut down Clippy? This stops both servers and wipes temp files.")) return;
    setShuttingDown(true);
    stopPolling();
    try {
      await fetch(`${API}/api/shutdown`, { method: "POST" });
    } catch {
      // Expected — the server may die before the response returns.
    }
  }, [stopPolling]);

  const isProcessing = status === "uploading" || status === "processing";

  if (shuttingDown) {
    return (
      <main className="flex-1 flex items-center justify-center px-4 py-12">
        <div className="w-full max-w-lg text-center space-y-3">
          <div className="flex items-center justify-center gap-2">
            <ClippyLogo className="h-8 w-8" />
            <h1 className="text-3xl font-bold tracking-tight text-neutral-100">Clippy</h1>
          </div>
          <p className="text-sm text-neutral-400">Clippy is shutting down.</p>
          <p className="text-xs text-neutral-600">You can close this tab.</p>
        </div>
      </main>
    );
  }

  return (
    <main className="flex-1 flex items-start justify-center px-4 py-12">
      <div className="w-full max-w-lg space-y-6">
        <div className="text-center space-y-2">
          <div className="flex items-center justify-center gap-2">
            <ClippyLogo className="h-8 w-8" />
            <h1 className="text-3xl font-bold tracking-tight text-neutral-100">
              Clippy
            </h1>
          </div>
          <p className="text-sm text-neutral-500">
            Clip the silent moments out of your footage, automatically.
          </p>
        </div>

        <Dropzone files={files} onFilesChange={setFiles} disabled={isProcessing} />

        <Controls
          threshold={threshold}
          startPadding={startPadding}
          endPadding={endPadding}
          minSilence={minSilence}
          onThresholdChange={setThreshold}
          onStartPaddingChange={setStartPadding}
          onEndPaddingChange={setEndPadding}
          onMinSilenceChange={setMinSilence}
          disabled={isProcessing}
        />

        {status === "idle" && (
          <button
            onClick={handleProcess}
            disabled={files.length === 0}
            className="w-full rounded-lg bg-violet-600 py-3 text-sm font-medium text-white transition-colors hover:bg-violet-500 disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {files.length > 1
              ? `Process & Trim ${files.length} Videos`
              : "Process Video"}
          </button>
        )}

        <StatusBar
          status={status}
          step={step}
          segments={segments}
          inputSizeMb={inputSizeMb}
          outputSizeMb={outputSizeMb}
          jobId={jobId}
          batch={batch}
          totalFiles={totalFiles}
          currentFile={currentFile}
          trimmedFiles={trimmedFiles}
          segmentCount={segmentCount}
          onReset={handleReset}
          onReprocess={handleReprocess}
        />

        <div className="pt-4 text-center">
          <button
            onClick={handleShutdown}
            disabled={isProcessing}
            className="text-xs text-neutral-600 hover:text-red-400 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
          >
            Shut down Clippy
          </button>
        </div>
      </div>
    </main>
  );
}
