// Mirrors backend/app/schemas.py

export type Phase = 'idle' | 'outdoor' | 'approach' | 'indoor' | 'arrived';

export enum Priority {
  INFO = 0,
  GUIDANCE = 1,
  WARNING = 2,
  CRITICAL = 3,
}

export interface Utterance {
  text: string;
  priority: Priority;
  category: 'hazard' | 'nav' | 'arrival' | 'scene' | 'answer' | 'system' | 'info';
  interrupt: boolean;
  haptic: 'none' | 'light' | 'warning' | 'success';
}

export interface ArrivalStatus {
  state: 'searching' | 'candidate' | 'confirmed';
  target: string | null;
  evidence: string[];
  confidence: number;
}

export interface Destination {
  raw: string;
  room?: string | null;
  sign_text?: string | null;
  target_kind?: string | null;
  place_name?: string | null;
  floor_hint?: number | null;
}

export interface Intent {
  name: string;
  target_kind?: string | null;
  text: string;
}

export interface AgentResponse {
  utterances: Utterance[];
  phase: Phase;
  step_index: number | null;
  arrival: ArrivalStatus | null;
  transcript: string | null;
  intent: Intent | null;
  destination: Destination | null;
  client_action: 'capture_frame' | 'capture_frame_read' | null;
  tool_trace?: { tool?: string; ms?: number; model_ms?: number; step?: number }[] | null;
  handled_by?: 'brain' | 'rules' | null;
}

export interface Hazard {
  type: string;
  severity: Priority;
  side: string;
  distance_m: number | null;
}

export interface FrameResponse extends AgentResponse {
  hazards: Hazard[];
  ocr: { text: string; kind: string }[];
  scene: { summary: string } | null;
  timings_ms: Record<string, number>;
}

export type FrameMode = 'auto' | 'describe' | 'read';

/** The parts of the server's session the phone needs for on-device course checks. */
export interface SessionSnapshot {
  phase: Phase;
  step_index: number;
  route: { instruction: string; distance_m: number; end: { lat: number; lng: number } }[];
  route_polyline: [number, number][];
  destination: (Destination & { place_location?: { lat: number; lng: number } | null }) | null;
  current_floor: number | null;
}
