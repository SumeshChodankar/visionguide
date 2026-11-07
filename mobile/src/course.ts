/**
 * Course checker: runs on the phone every 0.5 s, no network.
 * Compares the compass heading with the direction of the route ~15 m ahead and
 * your distance from the route line, and decides what (if anything) to say.
 */
import { bearing, clockFromRel, haversine, type LL, projectOnRoute, relAngle } from './geo';

export type CourseKind = 'ok' | 'turn' | 'back' | 'drift';

export interface CourseInput {
  now: number;
  pos: LL | null;
  accuracy: number | null;   // GPS accuracy, metres
  heading: number | null;    // compass, degrees
  walking: boolean;
  line: LL[];                // route polyline from the server
}

export interface CourseAdvice {
  kind: CourseKind;
  text: string;
  urgent: boolean;           // interrupt current speech
}

export interface CourseResult {
  advice: CourseAdvice | null;
  pulse: boolean;            // vibrate now (off course, between spoken corrections)
  status: string;            // short on-screen status
  kind: CourseKind | null;
}

export interface CourseState {
  kind: CourseKind | null;
  since: number;
  spokenAt: Partial<Record<CourseKind, number>>;
  corrected: boolean;        // we told them to correct; say "On track." once they do
  lastPulse: number;
}

export const newCourseState = (): CourseState => ({ kind: null, since: 0, spokenAt: {}, corrected: false, lastPulse: 0 });

const TURN_DEG = 45;
const BACK_DEG = 135;
const ONTRACK_DEG = 25;
const COOLDOWN: Record<CourseKind, number> = { ok: 0, turn: 6000, back: 5000, drift: 8000 };

export function checkCourse(st: CourseState, inp: CourseInput): CourseResult {
  const idle: CourseResult = { advice: null, pulse: false, status: '', kind: null };
  if (!inp.pos || inp.line.length < 2) return idle;
  const proj = projectOnRoute(inp.pos, inp.line, 15);
  if (!proj) return idle;
  if (proj.remaining < 10 || haversine(inp.pos, inp.line[inp.line.length - 1]) < 12) {
    st.kind = null;
    return { ...idle, status: 'arriving' };  // the server handles arrival at the building
  }
  if (inp.heading == null) return { ...idle, status: 'no compass' };

  const err = relAngle(bearing(inp.pos, proj.target), inp.heading);
  const abs = Math.abs(err);
  const clock = clockFromRel(err);
  const driftLimit = Math.max(15, (inp.accuracy ?? 10) * 1.5);
  const kind: CourseKind =
    abs > BACK_DEG ? 'back' : abs > TURN_DEG ? 'turn' : inp.walking && proj.offRoute > driftLimit ? 'drift' : 'ok';

  if (kind !== st.kind) {
    st.kind = kind;
    st.since = inp.now;
  }
  const status = `${kind === 'ok' ? 'on course' : kind === 'back' ? 'wrong way' : kind === 'turn' ? `turn ${err < 0 ? 'left' : 'right'}` : 'drifting'} · path ${clock} o'clock · ${Math.round(proj.offRoute)} m off`;

  if (kind === 'ok') {
    if (st.corrected && abs <= ONTRACK_DEG) {
      st.corrected = false;
      return { advice: { kind, text: 'On track.', urgent: false }, pulse: false, status, kind };
    }
    return { advice: null, pulse: false, status, kind };
  }

  const dwell = inp.walking ? 2000 : 4000;   // standing still and looking around shouldn't nag
  if (inp.now - st.since < dwell) return { advice: null, pulse: false, status, kind };

  let advice: CourseAdvice | null = null;
  if (inp.now - (st.spokenAt[kind] ?? 0) >= COOLDOWN[kind]) {
    st.spokenAt[kind] = inp.now;
    st.corrected = true;
    if (kind === 'back') {
      advice = { kind, text: `Wrong way. Turn around. Your path is at ${clock} o'clock.`, urgent: true };
    } else if (kind === 'turn') {
      advice = { kind, text: `Turn ${err < 0 ? 'left' : 'right'}. Your path is at ${clock} o'clock.`, urgent: false };
    } else {
      advice = {
        kind,
        text: `You're drifting off the route, ${Math.round(proj.offRoute)} meters to its ${proj.side > 0 ? 'right' : 'left'}. Your path is at ${clock} o'clock.`,
        urgent: false,
      };
    }
  }
  const pulse = !advice && inp.now - st.lastPulse >= 1500;
  if (pulse || advice) st.lastPulse = inp.now;
  return { advice, pulse, status, kind };
}
