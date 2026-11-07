import { Platform } from 'react-native';
import { API_URL, CONFIG } from './config';
import type { AgentResponse, FrameMode, FrameResponse, SessionSnapshot, Utterance } from './types';

async function request<T>(path: string, init: RequestInit = {}, timeoutMs = CONFIG.requestTimeoutMs): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(`${API_URL}${path}`, { ...init, signal: controller.signal });
    if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
    return (await res.json()) as T;
  } finally {
    clearTimeout(timer);
  }
}

const json = (body: unknown): RequestInit => ({
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
});

// React Native's FormData accepts { uri, name, type } file descriptors. Browsers don't:
// they would send the string "[object Object]", so on web we fetch the data:/blob: URI into a real Blob.
async function appendFile(form: FormData, field: string, uri: string, name: string, type: string) {
  if (Platform.OS === 'web') {
    const blob = await (await fetch(uri)).blob();
    form.append(field, new Blob([blob], { type: blob.type || type }), name);
  } else {
    form.append(field, { uri, name, type } as unknown as Blob);
  }
}

export const api = {
  health: () => request<Record<string, unknown>>('/health'),

  createSession: () =>
    request<{ session_id: string; utterances: Utterance[] }>('/v1/sessions', { method: 'POST' }),

  position: (sid: string, p: { lat: number; lng: number; heading: number | null; accuracy_m: number | null;
    floor_change?: number | null }) =>
    request<AgentResponse>(`/v1/sessions/${sid}/position`, json(p)),

  /** Full session state; used for the route geometry the phone checks your heading against. */
  session: (sid: string) => request<SessionSnapshot>(`/v1/sessions/${sid}`),

  /** YOLO-only safety pass (~0.1-0.3 s on the server), sent about once a second while live. */
  fastFrame: async (sid: string, imageUri: string) => {
    const form = new FormData();
    await appendFile(form, 'image', imageUri, 'fast.jpg', 'image/jpeg');
    return request<FrameResponse>(`/v1/sessions/${sid}/fast_frame`, { method: 'POST', body: form }, 4000);
  },

  command: (sid: string, text: string) =>
    request<AgentResponse>(`/v1/sessions/${sid}/command`, json({ text }), CONFIG.voiceTimeoutMs),

  /** Typed command + optional camera frame (same server path as voice, without Whisper). */
  ask: async (sid: string, text: string, imageUri?: string | null) => {
    const form = new FormData();
    form.append('text', text);
    if (imageUri) await appendFile(form, 'image', imageUri, 'frame.jpg', 'image/jpeg');
    return request<AgentResponse>(`/v1/sessions/${sid}/ask`, { method: 'POST', body: form }, CONFIG.voiceTimeoutMs);
  },

  chat: (sid: string, message: string) => request<AgentResponse>(`/v1/sessions/${sid}/chat`, json({ message })),

  voice: async (sid: string, audioUri: string, imageUri?: string | null) => {
    const form = new FormData();
    const ext = audioUri.split('.').pop() ?? 'm4a';
    await appendFile(form, 'audio', audioUri, `speech.${ext}`, ext === 'wav' ? 'audio/wav' : 'audio/m4a');
    // The brain's look_around tool analyses this frame: what the user faces while speaking.
    if (imageUri) await appendFile(form, 'image', imageUri, 'frame.jpg', 'image/jpeg');
    return request<AgentResponse>(`/v1/sessions/${sid}/voice`, { method: 'POST', body: form }, CONFIG.voiceTimeoutMs);
  },

  frame: async (sid: string, imageUri: string, mode: FrameMode, pos?: { lat: number; lng: number; heading: number | null }) => {
    const form = new FormData();
    await appendFile(form, 'image', imageUri, 'frame.jpg', 'image/jpeg');
    form.append('mode', mode);
    if (pos) {
      form.append('lat', String(pos.lat));
      form.append('lng', String(pos.lng));
      if (pos.heading != null) form.append('heading', String(pos.heading));
    }
    return request<FrameResponse>(`/v1/sessions/${sid}/frame`, { method: 'POST', body: form });
  },
};
