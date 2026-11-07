import * as Location from 'expo-location';
import { Accelerometer, Barometer, Gyroscope, LightSensor, Pedometer } from 'expo-sensors';
import { useEffect, useRef, useState } from 'react';
import { Platform } from 'react-native';

import { blendHeading } from '../geo';

/**
 * Every motion sensor the phone has, in one place:
 *
 *   compass      fused heading (rotation-vector sensor on Android), smoothed      → which way you face
 *   pedometer    step events                                                     → walking or standing
 *   accelerometer movement energy (fallback when the step counter is unavailable)
 *   gyroscope    fast rotations                                                  → "you just turned": look again now
 *   light        illuminance (Android)                                          → too dark for the camera
 *   barometer    air pressure (≈0.12 hPa per metre)                             → went up / down a floor
 *
 * Values live in a ref (read at any moment without re-rendering); a small
 * summary is mirrored into state ~2x per second for the on-screen status line.
 */
export interface SensorState {
  heading: number | null;
  headingAccuracy: number | null;
  walking: boolean;
  lastStepAt: number;
  steps: number;
  lux: number | null;
  pressure: number | null;
  /** floors moved since the last time someone consumed it (see takeFloorChange) */
  pendingFloorChange: number;
}

const HPA_PER_FLOOR = 0.4;    // ~3.3 m storey height
const WALK_WINDOW_MS = 2500;  // a step within this window = walking
const TURN_RATE = 1.2;        // rad/s (~70°/s): a deliberate turn, not sway

export function useSensors(opts: { onTurn?: () => void; onFloorChange?: (change: number) => void; onDark?: () => void }) {
  const s = useRef<SensorState>({
    heading: null, headingAccuracy: null, walking: false, lastStepAt: 0, steps: 0, lux: null, pressure: null,
    pendingFloorChange: 0,
  });
  const cb = useRef(opts);
  cb.current = opts;
  const [summary, setSummary] = useState({ heading: null as number | null, walking: false, lux: null as number | null,
    baro: false, steps: 0 });

  useEffect(() => {
    if (Platform.OS === 'web') return; // phones only (Expo Go)
    const subs: { remove: () => void }[] = [];
    let cancelled = false;
    let lastTurn = 0;
    let darkSince = 0;
    let darkWarned = false;
    let baseline: number | null = null;
    const pressures: number[] = [];
    let accelEnergy = 0;

    (async () => {
      // Compass: fused heading from Android's rotation-vector sensor.
      try {
        const { status } = await Location.requestForegroundPermissionsAsync();
        if (status === 'granted' && !cancelled) {
          subs.push(await Location.watchHeadingAsync((h) => {
            const raw = h.trueHeading >= 0 ? h.trueHeading : h.magHeading;
            if (raw == null || raw < 0) return;
            s.current.heading = blendHeading(s.current.heading, raw, 0.35);
            s.current.headingAccuracy = h.accuracy ?? null;
          }));
        }
      } catch {}

      // Step counter: the most reliable "am I walking" signal.
      try {
        if (await Pedometer.isAvailableAsync()) {
          const perm = await Pedometer.requestPermissionsAsync().catch(() => ({ granted: true }));
          if (perm.granted && !cancelled) {
            let last = 0;
            subs.push(Pedometer.watchStepCount(({ steps }) => {
              if (steps !== last) {
                s.current.lastStepAt = Date.now();
                s.current.steps += Math.max(0, steps - last);
                last = steps;
              }
            }));
          }
        }
      } catch {}

      // Accelerometer: movement energy, a fallback for phones without a step counter.
      try {
        if (await Accelerometer.isAvailableAsync()) {
          Accelerometer.setUpdateInterval(100);
          subs.push(Accelerometer.addListener(({ x, y, z }) => {
            const g = Math.sqrt(x * x + y * y + z * z);
            accelEnergy = 0.9 * accelEnergy + 0.1 * Math.abs(g - 1);   // smoothed deviation from 1 g
          }));
        }
      } catch {}

      // Gyroscope: a fast yaw turn means the view changed - look again immediately.
      try {
        if (await Gyroscope.isAvailableAsync()) {
          Gyroscope.setUpdateInterval(100);
          subs.push(Gyroscope.addListener(({ y, z }) => {
            // Upright phone (camera forward) turns around its y axis; flat phone around z.
            const yaw = Math.max(Math.abs(y), Math.abs(z));
            const now = Date.now();
            if (yaw > TURN_RATE && now - lastTurn > 1500) {
              lastTurn = now;
              cb.current.onTurn?.();
            }
          }));
        }
      } catch {}

      // Light sensor (Android): warn once if it's too dark for the camera to see.
      try {
        if (await LightSensor.isAvailableAsync()) {
          LightSensor.setUpdateInterval(1000);
          subs.push(LightSensor.addListener(({ illuminance }) => {
            s.current.lux = illuminance;
            const now = Date.now();
            if (illuminance < 8) {
              darkSince = darkSince || now;
              if (!darkWarned && now - darkSince > 5000) {
                darkWarned = true;
                cb.current.onDark?.();
              }
            } else if (illuminance > 30) {
              darkSince = 0;
              darkWarned = false;
            }
          }));
        }
      } catch {}

      // Barometer: floor changes in lifts and on stairs.
      try {
        if (await Barometer.isAvailableAsync()) {
          Barometer.setUpdateInterval(250);
          subs.push(Barometer.addListener(({ pressure }) => {
            pressures.push(pressure);
            if (pressures.length > 8) pressures.shift();           // ~2 s moving average
            const avg = pressures.reduce((a, b) => a + b, 0) / pressures.length;
            s.current.pressure = avg;
            if (baseline == null) {
              if (pressures.length >= 8) baseline = avg;
              return;
            }
            const floors = Math.trunc((baseline - avg) / HPA_PER_FLOOR);  // pressure falls going up
            if (Math.abs(baseline - avg) < 0.1) {
              baseline += 0.002 * (avg - baseline);  // follow slow weather drift (~2 min) while level
            }
            if (floors !== 0) {
              baseline -= floors * HPA_PER_FLOOR;                   // re-base by whole floors
              s.current.pendingFloorChange += floors;
              cb.current.onFloorChange?.(floors);
            }
          }));
        }
      } catch {}
    })();

    const tick = setInterval(() => {
      const now = Date.now();
      const stepping = now - s.current.lastStepAt < WALK_WINDOW_MS;
      s.current.walking = stepping || (s.current.lastStepAt === 0 && accelEnergy > 0.12);
      setSummary({
        heading: s.current.heading == null ? null : Math.round(s.current.heading),
        walking: s.current.walking, lux: s.current.lux == null ? null : Math.round(s.current.lux),
        baro: s.current.pressure != null, steps: s.current.steps,
      });
    }, 500);

    return () => {
      cancelled = true;
      clearInterval(tick);
      subs.forEach((x) => x.remove());
    };
  }, []);

  /** Floors moved since the last call (sent once with the next position update). */
  const takeFloorChange = () => {
    const n = s.current.pendingFloorChange;
    s.current.pendingFloorChange = 0;
    return n;
  };

  return { sensors: s, summary, takeFloorChange };
}
