import type { CameraView } from 'expo-camera';
import { ImageManipulator, SaveFormat } from 'expo-image-manipulator';
import { type RefObject, useCallback, useEffect, useRef, useState } from 'react';

import { api } from '../api';
import { CONFIG } from '../config';
import type { FrameMode, FrameResponse } from '../types';
import type { Fix } from './useLocationTracking';

/**
 * Live camera, two speeds:
 *
 *   fast  every ~1 s   small frame → YOLO only (~0.1-0.3 s): obstacles, vehicles, things getting closer
 *   full  every ~5 s   larger frame → Gemini + OCR + YOLO (5-8 s): signs, rooms, doors, scene; runs in the
 *                       background and never holds up the fast loop
 *
 * The camera is used by one capture at a time (a promise queue); network requests overlap freely.
 * `lookNow()` (called when the gyroscope sees you turn) schedules a fast look immediately
 * and brings the next full look forward.
 */
export function useSceneLoop(
  sessionId: string | null,
  camera: RefObject<CameraView | null>,
  fix: RefObject<Fix | null>,
  enabled: boolean,
  onResponse: (r: FrameResponse) => void,
) {
  const [lastLatency, setLastLatency] = useState<number | null>(null);
  const [fastLatency, setFastLatency] = useState<number | null>(null);
  const cameraQueue = useRef<Promise<unknown>>(Promise.resolve()); // serialises camera captures
  const fastBusy = useRef(false);    // fast request in flight
  const fullBusy = useRef(false);    // full request in flight
  const lastFull = useRef(0);
  const wakeFast = useRef<(() => void) | null>(null);
  const onResponseRef = useRef(onResponse);
  onResponseRef.current = onResponse;

  /** Take a photo and shrink it. Returns a local file URI (no upload). Captures run one at a time. */
  const grab = useCallback(
    (width: number, compress: number): Promise<string | null> => {
      const run = async () => {
        if (!camera.current) return null;
        const photo = await camera.current.takePictureAsync({ quality: 0.5, skipProcessing: true, shutterSound: false });
        if (!photo?.uri) return null;
        const ref = await ImageManipulator.manipulate(photo.uri).resize({ width }).renderAsync();
        const small = await ref.saveAsync({ compress, format: SaveFormat.JPEG });
        return small.uri;
      };
      const next = cameraQueue.current.then(run, run);
      cameraQueue.current = next.catch(() => null);
      return next;
    },
    [camera],
  );

  const pos = () =>
    fix.current ? { lat: fix.current.lat, lng: fix.current.lng, heading: fix.current.heading } : undefined;

  /** Full look (Gemini + OCR + YOLO). Fire-and-forget from the loop's point of view. */
  const full = useCallback(
    async (mode: FrameMode = 'auto') => {
      if (!sessionId || fullBusy.current) return;
      fullBusy.current = true;
      lastFull.current = Date.now();
      const t0 = Date.now();
      try {
        const uri = await grab(CONFIG.frameWidth, CONFIG.frameCompress);
        if (!uri) return;
        const r = await api.frame(sessionId, uri, mode, pos());
        setLastLatency(Date.now() - t0);
        onResponseRef.current(r);
      } catch (e) {
        console.warn('full frame failed', e);
      } finally {
        fullBusy.current = false;
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [sessionId, grab],
  );

  /** Fast look (YOLO only). */
  const fast = useCallback(async () => {
    if (!sessionId || fastBusy.current) return;
    fastBusy.current = true;
    const t0 = Date.now();
    try {
      const uri = await grab(CONFIG.fastFrameWidth, 0.5);
      if (!uri) return;
      const r = await api.fastFrame(sessionId, uri);
      setFastLatency(Date.now() - t0);
      onResponseRef.current(r);
    } catch {
      // a dropped fast frame is replaced a second later
    } finally {
      fastBusy.current = false;
    }
  }, [sessionId, grab]);

  useEffect(() => {
    if (!enabled || !sessionId) return;
    let stopped = false;
    let timer: ReturnType<typeof setTimeout>;
    const tick = async () => {
      if (stopped) return;
      const started = Date.now();
      if (Date.now() - lastFull.current >= CONFIG.sceneIntervalMs && !fullBusy.current) {
        full('auto'); // background: not awaited
      }
      await fast();
      if (stopped) return;
      timer = setTimeout(tick, Math.max(150, CONFIG.fastIntervalMs - (Date.now() - started)));
    };
    wakeFast.current = () => {
      clearTimeout(timer);
      lastFull.current = Math.min(lastFull.current, Date.now() - CONFIG.sceneIntervalMs + 1500); // full look soon
      timer = setTimeout(tick, 0);
    };
    timer = setTimeout(tick, 600);
    return () => {
      stopped = true;
      wakeFast.current = null;
      clearTimeout(timer);
    };
  }, [enabled, sessionId, fast, full]);

  /** The view just changed (you turned): look again right away. */
  const lookNow = useCallback(() => wakeFast.current?.(), []);

  /** Describe / read requests: wait for any in-flight full look, then do one in that mode. */
  const captureNow = useCallback(
    async (mode: FrameMode) => {
      for (let i = 0; i < 80 && fullBusy.current; i++) await new Promise((r) => setTimeout(r, 100));
      await full(mode);
    },
    [full],
  );

  /** A frame to attach to a voice command, so the brain sees what the user is facing. */
  const snapshot = useCallback(async (): Promise<string | null> => {
    try {
      return await grab(CONFIG.frameWidth, CONFIG.frameCompress);
    } catch {
      return null;
    }
  }, [grab]);

  return { captureNow, snapshot, lookNow, lastLatency, fastLatency };
}
