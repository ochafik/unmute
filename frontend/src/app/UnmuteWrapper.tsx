"use client";
/**
 * Wrapper component that handles switching between Local and OpenAI modes.
 *
 * - Local mode: Uses Kyutai STT/TTS services (Opus audio)
 * - OpenAI mode: Uses OpenAI Realtime API (PCM16 audio)
 */
import { useEffect, useState } from "react";
import Unmute from "./Unmute";
import UnmuteOpenAI from "./UnmuteOpenAI";
import CouldNotConnect, { HealthStatus } from "./CouldNotConnect";
import { useBackendServerUrl } from "./useBackendServerUrl";

type Mode = "local" | "openai";

const UnmuteWrapper = () => {
  const backendServerUrl = useBackendServerUrl();
  const [healthStatus, setHealthStatus] = useState<HealthStatus | null>(null);
  const [mode, setMode] = useState<Mode>("local");

  // Check backend health and determine available modes
  useEffect(() => {
    if (!backendServerUrl) return;

    const checkHealth = async () => {
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 3000);

        const response = await fetch(`${backendServerUrl}/v1/health`, {
          signal: controller.signal,
        });

        clearTimeout(timeoutId);
        if (!response.ok) {
          setHealthStatus({
            connected: "yes_request_fail",
            ok: false,
          });
          return;
        }

        const data = await response.json();
        data["connected"] = "yes_request_ok";
        setHealthStatus(data);

        // If local mode is not available but OpenAI is, switch to OpenAI
        if (!data.ok && data.openai_realtime_available) {
          setMode("openai");
        }
      } catch {
        setHealthStatus({
          connected: "no",
          ok: false,
        });
      }
    };

    checkHealth();
  }, [backendServerUrl]);

  // Loading state
  if (!healthStatus || !backendServerUrl) {
    return (
      <div className="flex flex-col gap-4 items-center justify-center min-h-screen bg-background text-white">
        <h1 className="text-xl mb-4">Loading...</h1>
      </div>
    );
  }

  // Neither mode available
  const localAvailable = healthStatus.ok;
  const openaiAvailable = healthStatus.openai_realtime_available || false;

  if (!localAvailable && !openaiAvailable) {
    return <CouldNotConnect healthStatus={healthStatus} />;
  }

  // Mode selector (only show if both modes are available)
  const showModeSelector = localAvailable && openaiAvailable;

  // Render based on selected mode
  if (mode === "openai" && openaiAvailable) {
    return (
      <UnmuteOpenAI
        backendServerUrl={backendServerUrl}
        healthStatus={healthStatus}
        onSwitchToLocal={localAvailable ? () => setMode("local") : undefined}
      />
    );
  }

  // Default to local mode with mode selector
  return (
    <div className="relative">
      {showModeSelector && (
        <div className="absolute top-4 left-4 z-20 flex items-center gap-2">
          <span className="px-2 py-1 bg-blue-600 text-white text-xs rounded">
            Local Mode
          </span>
          <button
            onClick={() => setMode("openai")}
            className="px-2 py-1 bg-green-600 hover:bg-green-500 text-white text-xs rounded"
          >
            Switch to OpenAI
          </button>
        </div>
      )}
      <Unmute />
    </div>
  );
};

export default UnmuteWrapper;
