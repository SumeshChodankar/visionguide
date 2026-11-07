import Constants from 'expo-constants';
import { Platform } from 'react-native';

/**
 * Where the backend is.
 *  1. EXPO_PUBLIC_API_URL in mobile/.env, if set (e.g. a deployed or tunnelled HTTPS URL).
 *  2. Otherwise, in development: the laptop Expo Go loaded the app from, on port 8001. This follows the
 *     laptop when it changes networks (UCLGuest, hotspot...) with no editing.
 */
const BACKEND_PORT = 8001;

function devServerHost(): string | null {
  const hostUri = Constants.expoConfig?.hostUri ?? (Constants as any).expoGoConfig?.debuggerHost ?? null;
  if (hostUri) return String(hostUri).split(':')[0];
  if (Platform.OS === 'web' && typeof window !== 'undefined') return window.location.hostname;
  return null;
}

const fromEnv = (process.env.EXPO_PUBLIC_API_URL ?? '').trim();
const host = devServerHost();
export const API_URL = (fromEnv || (host ? `http://${host}:${BACKEND_PORT}` : 'http://localhost:8001')).replace(/\/$/, '');

export const CONFIG = {
  sceneIntervalMs: Number(process.env.EXPO_PUBLIC_SCENE_INTERVAL_MS ?? 5000), // full Gemini + OCR look, in the background
  fastIntervalMs: 1000, // YOLO-only safety look while live: obstacles and things getting closer
  fastFrameWidth: 512,
  courseCheckMs: 500, // on-phone heading vs route check (no network)
  frameWidth: 960, // px sent to the server: enough for OCR on room plates, small enough for fast upload
  frameCompress: 0.6,
  positionIntervalMs: 1000,
  maxRecordMs: 8000,
  speechRate: 1.0, // natural pace: every word of the agent's answer is spoken as given
  repeatGapMs: 5000, // pause before the last answer is repeated (until the next instruction)
  wakeWord: 'Alex', // say "Alex, <instruction>" for hands-free control (web: Chrome / Edge)
  wakeWindowMs: 8000, // after "Alex" alone, how long to wait for the instruction
  requestTimeoutMs: 15000,
  voiceTimeoutMs: 50000, // brain turns chain several tool calls; keep above backend BRAIN_TIMEOUT_S
};
