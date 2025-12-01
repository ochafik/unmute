/**
 * PCM16 Audio Processor for OpenAI Realtime API mode.
 *
 * This processor captures audio from the microphone and sends it as base64-encoded
 * PCM16 at 24kHz mono (the native format for OpenAI Realtime API).
 *
 * For playback, it decodes base64 PCM16 and plays through the speakers.
 *
 * This avoids the Opus encoding/decoding overhead for optimal latency.
 */

import { useRef, useCallback } from "react";

const SAMPLE_RATE = 24000;
const BUFFER_SIZE = 4096; // ~170ms at 24kHz

export interface PCM16AudioProcessor {
  audioContext: AudioContext;
  inputAnalyser: AnalyserNode;
  outputAnalyser: AnalyserNode;
  mediaStreamDestination: MediaStreamAudioDestinationNode;
  scriptProcessor: ScriptProcessorNode;
  outputWorklet: AudioWorkletNode | null;
}

/**
 * Convert Float32 audio samples to base64-encoded PCM16.
 */
export const float32ToPCM16Base64 = (float32: Float32Array): string => {
  const int16 = new Int16Array(float32.length);
  for (let i = 0; i < float32.length; i++) {
    // Clamp to [-1, 1] and convert to int16
    const s = Math.max(-1, Math.min(1, float32[i]));
    int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  // Convert to base64
  const bytes = new Uint8Array(int16.buffer);
  let binary = "";
  for (let i = 0; i < bytes.length; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return window.btoa(binary);
};

/**
 * Convert base64-encoded PCM16 to Float32 audio samples.
 */
export const pcm16Base64ToFloat32 = (base64: string): Float32Array => {
  const binaryString = window.atob(base64);
  const bytes = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  const int16 = new Int16Array(bytes.buffer);
  const float32 = new Float32Array(int16.length);
  for (let i = 0; i < int16.length; i++) {
    float32[i] = int16[i] / 0x8000;
  }
  return float32;
};

/**
 * Resample audio from one sample rate to another.
 */
const resample = (
  input: Float32Array,
  inputRate: number,
  outputRate: number
): Float32Array => {
  if (inputRate === outputRate) {
    return input;
  }
  const ratio = inputRate / outputRate;
  const outputLength = Math.ceil(input.length / ratio);
  const output = new Float32Array(outputLength);

  for (let i = 0; i < outputLength; i++) {
    const srcIndex = i * ratio;
    const srcIndexFloor = Math.floor(srcIndex);
    const srcIndexCeil = Math.min(srcIndexFloor + 1, input.length - 1);
    const t = srcIndex - srcIndexFloor;
    // Linear interpolation
    output[i] = input[srcIndexFloor] * (1 - t) + input[srcIndexCeil] * t;
  }

  return output;
};

const getAudioWorkletNode = async (
  audioContext: AudioContext,
  name: string
) => {
  try {
    return new AudioWorkletNode(audioContext, name);
  } catch {
    await audioContext.audioWorklet.addModule(`/${name}.js`);
    return new AudioWorkletNode(audioContext, name, {});
  }
};

export const usePCM16AudioProcessor = (
  onPCM16Recorded: (pcm16Base64: string) => void
) => {
  const audioProcessorRef = useRef<PCM16AudioProcessor | null>(null);
  const playbackQueueRef = useRef<Float32Array[]>([]);
  const isPlayingRef = useRef(false);

  const setupAudio = useCallback(
    async (mediaStream: MediaStream) => {
      if (audioProcessorRef.current) return audioProcessorRef.current;

      // Create audio context at device sample rate, we'll resample to 24kHz
      const audioContext = new AudioContext();
      const deviceSampleRate = audioContext.sampleRate;

      // Set up output worklet for playback
      let outputWorklet: AudioWorkletNode | null = null;
      try {
        outputWorklet = await getAudioWorkletNode(
          audioContext,
          "audio-output-processor"
        );
        outputWorklet.connect(audioContext.destination);
      } catch (e) {
        console.warn("Could not load audio worklet, using fallback:", e);
      }

      // Set up media stream source
      const source = audioContext.createMediaStreamSource(mediaStream);

      // Input analyser for visualization
      const inputAnalyser = audioContext.createAnalyser();
      inputAnalyser.fftSize = 2048;
      source.connect(inputAnalyser);

      // Output analyser for visualization
      const outputAnalyser = audioContext.createAnalyser();
      outputAnalyser.fftSize = 2048;
      if (outputWorklet) {
        outputWorklet.connect(outputAnalyser);
      }

      // Media stream destination for recording canvas
      const mediaStreamDestination =
        audioContext.createMediaStreamDestination();
      if (outputWorklet) {
        outputWorklet.connect(mediaStreamDestination);
      }
      source.connect(mediaStreamDestination);

      // ScriptProcessor for capturing input audio
      // Note: ScriptProcessorNode is deprecated but still widely supported
      // AudioWorklet would be better but requires more setup
      const scriptProcessor = audioContext.createScriptProcessor(
        BUFFER_SIZE,
        1,
        1
      );

      // Accumulator for resampled audio to send in reasonable chunks
      let accumulatedSamples: Float32Array[] = [];
      let accumulatedLength = 0;
      const TARGET_CHUNK_SAMPLES = 2400; // 100ms at 24kHz

      scriptProcessor.onaudioprocess = (event) => {
        const inputData = event.inputBuffer.getChannelData(0);

        // Resample from device rate to 24kHz
        const resampled = resample(inputData, deviceSampleRate, SAMPLE_RATE);

        // Accumulate samples
        accumulatedSamples.push(resampled);
        accumulatedLength += resampled.length;

        // Send when we have enough samples
        if (accumulatedLength >= TARGET_CHUNK_SAMPLES) {
          // Concatenate accumulated samples
          const combined = new Float32Array(accumulatedLength);
          let offset = 0;
          for (const chunk of accumulatedSamples) {
            combined.set(chunk, offset);
            offset += chunk.length;
          }

          // Convert to PCM16 base64 and send
          const pcm16Base64 = float32ToPCM16Base64(combined);
          onPCM16Recorded(pcm16Base64);

          // Reset accumulator
          accumulatedSamples = [];
          accumulatedLength = 0;
        }
      };

      // Connect script processor (it needs to be connected to work)
      source.connect(scriptProcessor);
      scriptProcessor.connect(audioContext.destination);

      audioProcessorRef.current = {
        audioContext,
        inputAnalyser,
        outputAnalyser,
        mediaStreamDestination,
        scriptProcessor,
        outputWorklet,
      };

      // Resume audio context if suspended
      await audioContext.resume();

      return audioProcessorRef.current;
    },
    [onPCM16Recorded]
  );

  /**
   * Play PCM16 audio received from the server.
   */
  const playPCM16 = useCallback((pcm16Base64: string) => {
    if (!audioProcessorRef.current) return;

    const { audioContext, outputWorklet } = audioProcessorRef.current;
    const deviceSampleRate = audioContext.sampleRate;

    // Decode PCM16
    const float32_24k = pcm16Base64ToFloat32(pcm16Base64);

    // Resample from 24kHz to device rate
    const float32 = resample(float32_24k, SAMPLE_RATE, deviceSampleRate);

    if (outputWorklet) {
      // Use audio worklet for playback
      outputWorklet.port.postMessage({
        frame: float32,
        type: "audio",
        micDuration: 0,
      });
    } else {
      // Fallback: queue for playback with AudioBuffer
      playbackQueueRef.current.push(float32);
      if (!isPlayingRef.current) {
        playNextBuffer();
      }
    }
  }, []);

  const playNextBuffer = useCallback(() => {
    if (!audioProcessorRef.current) return;
    const { audioContext } = audioProcessorRef.current;

    if (playbackQueueRef.current.length === 0) {
      isPlayingRef.current = false;
      return;
    }

    isPlayingRef.current = true;
    const samples = playbackQueueRef.current.shift()!;

    const buffer = audioContext.createBuffer(
      1,
      samples.length,
      audioContext.sampleRate
    );
    buffer.getChannelData(0).set(samples);

    const source = audioContext.createBufferSource();
    source.buffer = buffer;
    source.connect(audioContext.destination);
    source.onended = playNextBuffer;
    source.start();
  }, []);

  const shutdownAudio = useCallback(() => {
    if (audioProcessorRef.current) {
      const { audioContext, scriptProcessor, outputWorklet } =
        audioProcessorRef.current;

      scriptProcessor.disconnect();
      outputWorklet?.disconnect();
      audioContext.close();

      audioProcessorRef.current = null;
      playbackQueueRef.current = [];
      isPlayingRef.current = false;
    }
  }, []);

  return {
    setupAudio,
    shutdownAudio,
    playPCM16,
    audioProcessor: audioProcessorRef,
  };
};
