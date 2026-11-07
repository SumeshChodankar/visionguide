import * as Location from 'expo-location';
import { type RefObject, useEffect, useRef, useState } from 'react';

import { api } from '../api';
import { CONFIG } from '../config';
import { speech } from '../speech';
import type { AgentResponse } from '../types';
import type { SensorState } from './useSensors';

export interface Fix {
  lat: number;
  lng: number;
  heading: number | null;
  accuracy: number | null;
  at: number;
}

/**
 * Streams GPS (every second) to the Navigation Agent, with the live compass heading from
 * useSensors and any barometer floor change, and speaks what comes back.
 */
export function useLocationTracking(
  sessionId: string | null,
  onResponse: (r: AgentResponse) => void,
  sensors?: RefObject<SensorState>,
  takeFloorChange?: () => number,
) {
  const fix = useRef<Fix | null>(null);
  const [granted, setGranted] = useState<boolean | null>(null);
  const lastSent = useRef(0);
  const inFlight = useRef(false);
  const onResponseRef = useRef(onResponse);
  onResponseRef.current = onResponse;
  const takeRef = useRef(takeFloorChange);
  takeRef.current = takeFloorChange;

  useEffect(() => {
    if (!sessionId) return;
    let posSub: Location.LocationSubscription | undefined;
    let cancelled = false;

    const heading = (gpsCourse: number | null | undefined) => {
      const compass = sensors?.current?.heading;
      if (compass != null) return compass;
      return gpsCourse != null && gpsCourse >= 0 ? gpsCourse : null; // web / no compass: course over ground
    };

    (async () => {
      const { status } = await Location.requestForegroundPermissionsAsync();
      if (cancelled) return;
      setGranted(status === 'granted');
      if (status !== 'granted') {
        speech.local('Location permission is off. Outdoor directions will not work until you allow it in Settings.');
        return;
      }
      posSub = await Location.watchPositionAsync(
        { accuracy: Location.Accuracy.BestForNavigation, timeInterval: CONFIG.positionIntervalMs, distanceInterval: 1 },
        async (loc) => {
          fix.current = {
            lat: loc.coords.latitude,
            lng: loc.coords.longitude,
            heading: heading(loc.coords.heading),
            accuracy: loc.coords.accuracy,
            at: Date.now(),
          };
          const now = Date.now();
          if (inFlight.current || now - lastSent.current < CONFIG.positionIntervalMs - 100) return;
          inFlight.current = true;
          lastSent.current = now;
          const floorChange = takeRef.current?.() ?? 0;
          try {
            const r = await api.position(sessionId, {
              lat: fix.current.lat,
              lng: fix.current.lng,
              heading: heading(loc.coords.heading),
              accuracy_m: fix.current.accuracy,
              floor_change: floorChange || null,
            });
            onResponseRef.current(r);
          } catch {
            // transient network errors are fine; the next fix retries (and carries the floor change)
            if (floorChange && sensors?.current) sensors.current.pendingFloorChange += floorChange;
          } finally {
            inFlight.current = false;
          }
        },
      );
    })();

    // Keep the compass heading in the shared fix fresh between GPS updates (~5x per second).
    const headTick = setInterval(() => {
      if (fix.current && sensors?.current?.heading != null) fix.current.heading = sensors.current.heading;
    }, 200);

    return () => {
      cancelled = true;
      clearInterval(headTick);
      posSub?.remove();
    };
  }, [sessionId, sensors]);

  return { fix, granted };
}
