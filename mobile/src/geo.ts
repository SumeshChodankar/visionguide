/** On-phone geometry for instant course checks (mirrors backend/app/utils/geo.py). */

export type LL = { lat: number; lng: number };
const R = 6371000;
const rad = (d: number) => (d * Math.PI) / 180;
const deg = (r: number) => (r * 180) / Math.PI;

export function haversine(a: LL, b: LL): number {
  const dp = rad(b.lat - a.lat);
  const dl = rad(b.lng - a.lng);
  const h = Math.sin(dp / 2) ** 2 + Math.cos(rad(a.lat)) * Math.cos(rad(b.lat)) * Math.sin(dl / 2) ** 2;
  return 2 * R * Math.asin(Math.min(1, Math.sqrt(h)));
}

/** Bearing from a to b, degrees clockwise from north. */
export function bearing(a: LL, b: LL): number {
  const dl = rad(b.lng - a.lng);
  const y = Math.sin(dl) * Math.cos(rad(b.lat));
  const x = Math.cos(rad(a.lat)) * Math.sin(rad(b.lat)) - Math.sin(rad(a.lat)) * Math.cos(rad(b.lat)) * Math.cos(dl);
  return (deg(Math.atan2(y, x)) + 360) % 360;
}

/** Signed angle from `heading` to `target` in -180..180 (+ = target is to the right). */
export function relAngle(target: number, heading: number): number {
  return ((target - heading + 540) % 360) - 180;
}

/** 12 = straight ahead, 3 = right, 6 = behind, 9 = left. */
export function clockFromRel(rel: number): number {
  const h = Math.round((((rel % 360) + 360) % 360) / 30) % 12;
  return h === 0 ? 12 : h;
}

/** Smooth compass readings (handles the 359°→0° wrap). */
export function blendHeading(prev: number | null, next: number, alpha = 0.3): number {
  if (prev == null) return next;
  return (prev + alpha * relAngle(next, prev) + 360) % 360;
}

// Local flat-earth projection, metres; fine over a few hundred metres.
function toXY(p: LL, ref: LL): [number, number] {
  return [rad(p.lng - ref.lng) * R * Math.cos(rad(ref.lat)), rad(p.lat - ref.lat) * R];
}
function fromXY([x, y]: [number, number], ref: LL): LL {
  return { lat: ref.lat + deg(y / R), lng: ref.lng + deg(x / (R * Math.cos(rad(ref.lat)))) };
}

export interface Projection {
  /** distance from the user to the route line, metres */
  offRoute: number;
  /** +1 if the user is to the right of the route's direction of travel, -1 if to the left */
  side: number;
  /** point on the route `lookAheadM` beyond the user's nearest point: where they should head */
  target: LL;
  /** metres of route left from the user's nearest point */
  remaining: number;
}

/** Project the user onto the route polyline and pick a look-ahead target to steer toward. */
export function projectOnRoute(user: LL, line: LL[], lookAheadM = 15): Projection | null {
  if (line.length < 2) return null;
  const P = toXY(user, user);
  let best = { d: Infinity, seg: 0, t: 0, side: 1 };
  for (let i = 0; i < line.length - 1; i++) {
    const A = toXY(line[i], user);
    const B = toXY(line[i + 1], user);
    const vx = B[0] - A[0];
    const vy = B[1] - A[1];
    const len2 = vx * vx + vy * vy || 1e-9;
    const t = Math.max(0, Math.min(1, ((P[0] - A[0]) * vx + (P[1] - A[1]) * vy) / len2));
    const cx = A[0] + t * vx;
    const cy = A[1] + t * vy;
    const d = Math.hypot(P[0] - cx, P[1] - cy);
    if (d < best.d) {
      const cross = vx * (P[1] - A[1]) - vy * (P[0] - A[0]); // >0: user left of travel direction
      best = { d, seg: i, t, side: cross > 0 ? -1 : 1 };
    }
  }
  // Walk forward along the line from the nearest point.
  let need = lookAheadM;
  let remaining = 0;
  let target: LL | null = null;
  for (let i = best.seg; i < line.length - 1; i++) {
    const A = toXY(line[i], user);
    const B = toXY(line[i + 1], user);
    const start: [number, number] = i === best.seg ? [A[0] + best.t * (B[0] - A[0]), A[1] + best.t * (B[1] - A[1])] : A;
    const segLen = Math.hypot(B[0] - start[0], B[1] - start[1]);
    remaining += segLen;
    if (!target) {
      if (segLen >= need) {
        const f = need / (segLen || 1);
        target = fromXY([start[0] + f * (B[0] - start[0]), start[1] + f * (B[1] - start[1])], user);
      } else {
        need -= segLen;
      }
    }
  }
  return { offRoute: best.d, side: best.side, target: target ?? line[line.length - 1], remaining };
}
