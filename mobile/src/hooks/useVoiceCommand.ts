import {
  RecordingPresets,
  requestRecordingPermissionsAsync,
  setAudioModeAsync,
  useAudioRecorder,
} from 'expo-audio';
import * as Haptics from 'expo-haptics';
import { useCallback, useEffect, useRef, useState } from 'react';

import { api } from '../api';
import { CONFIG } from '../config';
import { speech } from '../speech';
import type { AgentResponse } from '../types';

/**
 * Tap-to-talk (tap again to send, auto-sends after 8 s). Toggle rather than
 * press-and-hold because hold gestures are awkward under VoiceOver/TalkBack.
 */
export function useVoiceCommand(
  sessionId: string | null,
  onResponse: (r: AgentResponse) => void,
  snapshot?: () => Promise<string | null>,
) {
  const recorder = useAudioRecorder(RecordingPresets.HIGH_QUALITY);
  const [recording, setRecording] = useState(false);
  const [busy, setBusy] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const onResponseRef = useRef(onResponse);
  onResponseRef.current = onResponse;

  useEffect(() => {
    requestRecordingPermissionsAsync().then(({ granted }) => {
      if (!granted) speech.local('Microphone permission is off. Voice commands need it.');
    });
    setAudioModeAsync({ playsInSilentMode: true, allowsRecording: false });
  }, []);

  const stop = useCallback(async () => {
    if (timer.current) clearTimeout(timer.current);
    if (!recording) return;
    setRecording(false);
    setBusy(true);
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
    try {
      // Grab the camera frame in parallel with finalising the recording.
      const framePromise = snapshot ? snapshot().catch(() => null) : Promise.resolve(null);
      await recorder.stop();
      // Switch the iOS audio session back to playback so TTS uses the loudspeaker.
      await setAudioModeAsync({ playsInSilentMode: true, allowsRecording: false });
      speech.release();
      const uri = recorder.uri;
      if (!uri || !sessionId) throw new Error('no recording');
      speech.local('One moment.', 0);
      onResponseRef.current(await api.voice(sessionId, uri, await framePromise));
    } catch {
      speech.release();
      speech.local("Sorry, I couldn't process that. Please try again.");
    } finally {
      setBusy(false);
    }
  }, [recorder, recording, sessionId, snapshot]);

  const start = useCallback(async () => {
    if (recording || busy) return;
    await speech.hold();
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Heavy);
    await setAudioModeAsync({ playsInSilentMode: true, allowsRecording: true });
    await recorder.prepareToRecordAsync();
    recorder.record();
    setRecording(true);
  }, [recorder, recording, busy]);

  useEffect(() => {
    if (recording) timer.current = setTimeout(stop, CONFIG.maxRecordMs);
    return () => {
      if (timer.current) clearTimeout(timer.current);
    };
  }, [recording, stop]);

  const toggle = useCallback(() => (recording ? stop() : start()), [recording, start, stop]);

  return { recording, busy, toggle };
}
