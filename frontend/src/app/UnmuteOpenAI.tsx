"use client";
/**
 * OpenAI Realtime API mode for Unmute.
 *
 * This component provides a voice interface using OpenAI's Realtime API
 * instead of the local STT/TTS services. It uses PCM16 audio format
 * for optimal latency (no Opus encoding/decoding).
 */
import useWebSocket, { ReadyState } from "react-use-websocket";
import { useCallback, useEffect, useState } from "react";
import { useMicrophoneAccess } from "./useMicrophoneAccess";
import SlantedButton from "@/app/SlantedButton";
import { usePCM16AudioProcessor } from "./usePCM16AudioProcessor";
import useKeyboardShortcuts from "./useKeyboardShortcuts";
import { prettyPrintJson } from "pretty-print-json";
import PositionedAudioVisualizer from "./PositionedAudioVisualizer";
import UnmuteConfigurator, {
  DEFAULT_UNMUTE_CONFIG,
  UnmuteConfig,
} from "./UnmuteConfigurator";
import { HealthStatus } from "./CouldNotConnect";
import UnmuteHeader from "./UnmuteHeader";
import Subtitles from "./Subtitles";
import { ChatMessage, compressChatHistory } from "./chatHistory";
import useWakeLock from "./useWakeLock";
import ErrorMessages, { ErrorItem, makeErrorItem } from "./ErrorMessages";
import clsx from "clsx";

interface UnmuteOpenAIProps {
  backendServerUrl: URL;
  healthStatus: HealthStatus;
  onSwitchToLocal?: () => void;
}

const UnmuteOpenAI = ({
  backendServerUrl,
  healthStatus,
  onSwitchToLocal,
}: UnmuteOpenAIProps) => {
  const { isDevMode, showSubtitles } = useKeyboardShortcuts();
  const [debugDict, setDebugDict] = useState<object | null>(null);
  // Default config for OpenAI mode with an OpenAI voice
  const [unmuteConfig, setUnmuteConfig] = useState<UnmuteConfig>({
    ...DEFAULT_UNMUTE_CONFIG,
    voice: "alloy", // Default OpenAI voice
    voiceName: "Alloy",
  });
  const [rawChatHistory, setRawChatHistory] = useState<ChatMessage[]>([]);
  const chatHistory = compressChatHistory(rawChatHistory);

  const { microphoneAccess, askMicrophoneAccess } = useMicrophoneAccess();

  const [shouldConnect, setShouldConnect] = useState(false);
  const [errors, setErrors] = useState<ErrorItem[]>([]);

  // WebSocket URL for OpenAI mode
  const webSocketUrl = backendServerUrl.toString() + "/v1/realtime/openai";

  useWakeLock(shouldConnect);

  const { sendMessage, lastMessage, readyState } = useWebSocket(
    webSocketUrl,
    {
      protocols: ["realtime"],
    },
    shouldConnect
  );

  // Send microphone audio to the server as PCM16
  const onPCM16Recorded = useCallback(
    (pcm16Base64: string) => {
      sendMessage(
        JSON.stringify({
          type: "input_audio_buffer.append",
          audio: pcm16Base64,
        })
      );
    },
    [sendMessage]
  );

  const { setupAudio, shutdownAudio, playPCM16, audioProcessor } =
    usePCM16AudioProcessor(onPCM16Recorded);

  const onConnectButtonPress = async () => {
    if (!shouldConnect) {
      const mediaStream = await askMicrophoneAccess();
      if (mediaStream) {
        await setupAudio(mediaStream);
        setShouldConnect(true);
      }
    } else {
      setShouldConnect(false);
      shutdownAudio();
    }
  };

  // If the websocket connection is closed, shut down the audio processing
  useEffect(() => {
    if (readyState === ReadyState.CLOSING || readyState === ReadyState.CLOSED) {
      setShouldConnect(false);
      shutdownAudio();
    }
  }, [readyState, shutdownAudio]);

  // Handle incoming messages from the server
  useEffect(() => {
    if (lastMessage === null) return;

    const data = JSON.parse(lastMessage.data);

    if (data.type === "response.audio.delta") {
      // Play PCM16 audio directly
      playPCM16(data.delta);
    } else if (data.type === "error") {
      if (data.error.type === "warning") {
        console.warn(`Warning from server: ${data.error.message}`, data);
      } else {
        console.error(`Error from server: ${data.error.message}`, data);
        setErrors((prev) => [...prev, makeErrorItem(data.error.message)]);
      }
    } else if (
      data.type === "conversation.item.input_audio_transcription.delta" ||
      data.type === "conversation.item.input_audio_transcription.completed"
    ) {
      // Transcription of the user speech
      const transcript = data.delta || data.transcript;
      if (transcript) {
        setRawChatHistory((prev) => [
          ...prev,
          { role: "user", content: transcript },
        ]);
      }
    } else if (
      data.type === "response.text.delta" ||
      data.type === "response.audio_transcript.delta"
    ) {
      // Assistant's response text
      setRawChatHistory((prev) => [
        ...prev,
        { role: "assistant", content: " " + data.delta },
      ]);
    } else if (data.type === "unmute.additional_outputs") {
      setDebugDict(data.args?.debug_dict);
    } else {
      const ignoredTypes = [
        "session.created",
        "session.updated",
        "response.created",
        "response.done",
        "response.text.done",
        "response.audio.done",
        "response.audio_transcript.done",
        "response.output_item.added",
        "response.output_item.done",
        "response.content_part.added",
        "response.content_part.done",
        "input_audio_buffer.speech_stopped",
        "input_audio_buffer.speech_started",
        "input_audio_buffer.committed",
        "conversation.item.created",
        "rate_limits.updated",
      ];
      if (!ignoredTypes.includes(data.type)) {
        console.warn("Received unknown message:", data);
      }
    }
  }, [lastMessage, playPCM16]);

  // When we connect, send the initial config
  useEffect(() => {
    if (readyState !== ReadyState.OPEN) return;

    setRawChatHistory([]);
    sendMessage(
      JSON.stringify({
        type: "session.update",
        session: {
          instructions: unmuteConfig.instructions,
          voice: unmuteConfig.voice,
          allow_recording: false, // OpenAI mode doesn't support local recording
        },
      })
    );
  }, [unmuteConfig, readyState, sendMessage]);

  // Disconnect when the voice or instruction changes
  useEffect(() => {
    setShouldConnect(false);
    shutdownAudio();
  }, [shutdownAudio, unmuteConfig.voice, unmuteConfig.instructions]);

  return (
    <div className="w-full">
      <ErrorMessages errors={errors} setErrors={setErrors} />
      <div className="relative flex w-full min-h-screen flex-col text-white bg-background items-center">
        <header className="static md:absolute max-w-6xl px-3 md:px-8 right-0 flex justify-end z-10">
          <UnmuteHeader />
        </header>

        {/* OpenAI Mode Indicator */}
        <div className="absolute top-4 left-4 z-10 flex items-center gap-2">
          <span className="px-2 py-1 bg-green-600 text-white text-xs rounded">
            OpenAI Realtime
          </span>
          {onSwitchToLocal && (
            <button
              onClick={onSwitchToLocal}
              className="px-2 py-1 bg-gray-600 hover:bg-gray-500 text-white text-xs rounded"
            >
              Switch to Local
            </button>
          )}
        </div>

        <div
          className={clsx(
            "w-full h-auto min-h-75",
            "flex flex-row-reverse md:flex-row items-center justify-center grow",
            "-mt-10 md:mt-0 mb-10 md:mb-0 md:-mr-4"
          )}
        >
          <PositionedAudioVisualizer
            chatHistory={chatHistory}
            role={"assistant"}
            analyserNode={audioProcessor.current?.outputAnalyser || null}
            onCircleClick={onConnectButtonPress}
            isConnected={shouldConnect}
          />
          <PositionedAudioVisualizer
            chatHistory={chatHistory}
            role={"user"}
            analyserNode={audioProcessor.current?.inputAnalyser || null}
            isConnected={shouldConnect}
          />
        </div>
        {showSubtitles && <Subtitles chatHistory={chatHistory} />}
        <UnmuteConfigurator
          backendServerUrl={backendServerUrl}
          config={unmuteConfig}
          setConfig={setUnmuteConfig}
          voiceCloningUp={false} // Voice cloning not supported in OpenAI mode
          openaiMode={true}
        />
        <div className="w-full flex flex-col-reverse md:flex-row items-center justify-center px-3 gap-3 my-6">
          <SlantedButton
            onClick={onConnectButtonPress}
            kind={shouldConnect ? "secondary" : "primary"}
            extraClasses="w-full max-w-96"
          >
            {shouldConnect ? "disconnect" : "connect"}
          </SlantedButton>
          {microphoneAccess === "refused" && (
            <div className="text-red">
              {"You'll need to allow microphone access to use the demo. " +
                "Please check your browser settings."}
            </div>
          )}
        </div>
      </div>
      {isDevMode && (
        <div>
          <div className="text-xs w-full overflow-auto">
            <pre
              className="whitespace-pre-wrap break-words"
              dangerouslySetInnerHTML={{
                __html: prettyPrintJson.toHtml(debugDict),
              }}
            ></pre>
          </div>
          <div>Subtitles: press S. Dev mode: press D.</div>
        </div>
      )}
    </div>
  );
};

export default UnmuteOpenAI;
