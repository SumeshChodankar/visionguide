import * as Haptics from 'expo-haptics';
import { type RefObject, useEffect, useRef, useState } from 'react';
import { Platform } from 'react-native';

import { api } from '../api';
import { CONFIG } from '../config';
import { checkCourse, newCourseState } from '../course';
import type { LL } from '../geo';
import { speech } from '../speech';
import { Priority, type Phase } from '../types';
import type { Fix } from './useLocationTracking';
import type { SensorState } from './useSensors';

/**
 * Live course keeping while walking a route: every 0.5 s, on the phone, compares the
 * compass heading with the route and speaks corrections at once ("Turn left. Your
 * path is at 10 o'clock.", "Wrong way. Turn around."), with a vibration pulse while
 * off course. The route geometry is fetched from the server when navigation starts
 * and whenever the step changes or the route is recalculated.
 */
export function useCourseGuide(
  sessionId: string | null,
  phase: Phase,
  stepIndex: number | null,
  fix: RefObject<Fix | null>,
  sensors: RefObject<SensorState>,
) {
  const line = useRef<LL[]>([]);
  const state = useRef(newCourseState());
  const calibrateWarnedAt = useRef(0);
  const [status, setStatus] = useState('');

  // Route geometry: on entering the outdoor phase, on every step change (also covers re-routes).
  useEffect(() => {
    if (!sessionId || phase !== 'outdoor') {
      line.current = [];
      setStatus('');
      return;
    }
    let alive = true;
    api.session(sessionId).then(
      (s) => {
        if (alive) line.current = (s.route_polyline ?? []).map(([lat, lng]) => ({ lat, lng }));
      },
      () => {},
    );
    return () => {
      alive = false;
    };
  }, [sessionId, phase, stepIndex]);

  useEffect(() => {
    if (phase !== 'outdoor') return;
    state.current = newCourseState();
    const t = setInterval(() => {
      const f = fix.current;
      const sn = sensors.current;
      if (!f || !sn || line.current.length < 2) return;
      // Stale GPS (no fix for 10 s, e.g. under cover) gives wrong answers: stay quiet.
      if (Date.now() - f.at > 10000) {
        setStatus('waiting for GPS');
        return;
      }
      const r = checkCourse(state.current, {
        now: Date.now(), pos: { lat: f.lat, lng: f.lng }, accuracy: f.accuracy,
        heading: sn.heading, walking: sn.walking, line: line.current,
      });
      setStatus(r.status);
      if (r.advice) {
        speech.say([{
          text: r.advice.text,
          priority: r.advice.kind === 'ok' ? Priority.GUIDANCE : Priority.WARNING,
          category: 'nav',
          interrupt: r.advice.urgent,
          haptic: r.advice.kind === 'ok' ? 'success' : 'warning',
        }]);
      } else if (r.pulse) {
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium); // "still off course"
      }
      // Android reports compass calibration 0-3; 0-1 means headings can be badly off.
      const acc = sn.headingAccuracy;
      if (Platform.OS === 'android' && acc != null && acc <= 1 && Date.now() - calibrateWarnedAt.current > 300000) {
        calibrateWarnedAt.current = Date.now();
        speech.local('Your compass needs calibrating. Wave the phone in a figure eight a few times.', Priority.GUIDANCE);
      }
    }, CONFIG.courseCheckMs);
    return () => clearInterval(t);
  }, [phase, fix, sensors]);

  return { status };
}
