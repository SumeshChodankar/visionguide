import { useCallback, useEffect, useRef, useState } from 'react';
import { Platform } from 'react-native';

import { CONFIG } from '../config';

/**
 * Hands-free control: listens continuously and reacts to the wake word ("Alex").
 *
 *   "Alex, where am I?"          → command "where am I?" in one breath
 *   "Alex" … (beep) … "read the sign"  → command on the next phrase, within wakeWindowMs
 *
 * Uses the browser's built-in speech recognition (Chrome / Edge), so the transcript
 * arrives as text and no server-side speech-to-text is needed. Other speech, including
 * the app's own voice, is ignored unless it follows the wake word. Not available in
 * Expo Go on phones (needs a development build); there the tap-to-talk button is used.
 */

// Recognisers often hear "Alex" as Alec, Alexa, Alix, Aleks...
const WAKE = /\b(?:hey|ok|okay)?\s*(?:alex|alexa|alec|aleks|alix|alexis|elix|allex)\b[\s,.!?:;-]*/i;

type Recognition = {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  start(): void;
  stop(): void;
  abort(): void;
  onresult: ((e: any) => void) | null;
  onend: (() => void) | null;
  onerror: ((e: any) => void) | null;
};

function recognizerClass(): (new () => Recognition) | null {
  if (Platform.OS !== 'web' || typeof window === 'undefined') return null;
  const w = window as any;
  return w.SpeechRecognition || w.webkitSpeechRecognition || null;
}

export function useWakeWord(handlers: {
  onWake: () => void;
  onCommand: (text: string) => void;
  onError?: (message: string) => void;
}) {
  const supported = recognizerClass() !== null;
  const [listening, setListening] = useState(false);
  const [armed, setArmed] = useState(false);
  const [heard, setHeard] = useState<string | null>(null);

  const cb = useRef(handlers);
  cb.current = handlers;
  const rec = useRef<Recognition | null>(null);
  const want = useRef(false);
  const armedRef = useRef(false);
  const armTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const wokeInPhrase = useRef(false);

  const disarm = useCallback(() => {
    armedRef.current = false;
    setArmed(false);
    if (armTimer.current) clearTimeout(armTimer.current);
  }, []);

  /** Wait for the next phrase as a command (after "Alex", or after tapping the big button). */
  const arm = useCallback(() => {
    armedRef.current = true;
    setArmed(true);
    if (armTimer.current) clearTimeout(armTimer.current);
    armTimer.current = setTimeout(disarm, CONFIG.wakeWindowMs);
  }, [disarm]);

  const onFinal = useCallback(
    (text: string) => {
      const m = WAKE.exec(text);
      if (m) {
        if (!wokeInPhrase.current) cb.current.onWake();
        const command = text.slice(m.index + m[0].length).trim();
        if (command.replace(/[^a-z0-9]/gi, '').length >= 2) {
          disarm();
          cb.current.onCommand(command);
        } else {
          arm(); // "Alex" alone: the instruction comes in the next phrase
        }
      } else if (armedRef.current && text.trim()) {
        disarm();
        cb.current.onCommand(text.trim());
      }
      wokeInPhrase.current = false;
    },
    [arm, disarm],
  );

  const start = useCallback(() => {
    const Cls = recognizerClass();
    if (!Cls) return false;
    want.current = true;
    if (rec.current) return true;
    const r = new Cls();
    r.continuous = true;
    r.interimResults = true;
    r.lang = (typeof navigator !== 'undefined' && navigator.language) || 'en-US';
    r.onresult = (e: any) => {
      for (let i = e.resultIndex; i < e.results.length; i++) {
        const res = e.results[i];
        const text: string = res[0]?.transcript ?? '';
        if (res.isFinal) {
          setHeard(null);
          onFinal(text);
        } else {
          setHeard(text);
          // React the moment "Alex" is heard (silence the app) instead of waiting for the phrase to end.
          if (!wokeInPhrase.current && WAKE.test(text)) {
            wokeInPhrase.current = true;
            cb.current.onWake();
            arm();
          }
        }
      }
    };
    r.onerror = (e: any) => {
      if (e?.error === 'not-allowed' || e?.error === 'service-not-allowed') {
        want.current = false;
        cb.current.onError?.('Microphone access is blocked. Allow the microphone in the browser to use “Alex”.');
      }
      // no-speech / aborted / network: onend fires next and we restart
    };
    r.onend = () => {
      rec.current = null;
      if (want.current) setTimeout(() => want.current && start(), 300); // Chrome ends sessions after silence
      else setListening(false);
    };
    try {
      r.start();
      rec.current = r;
      setListening(true);
      return true;
    } catch {
      return false;
    }
  }, [arm, onFinal]);

  const stop = useCallback(() => {
    want.current = false;
    disarm();
    rec.current?.abort();
    rec.current = null;
    setListening(false);
  }, [disarm]);

  useEffect(() => () => stop(), [stop]);

  return { supported, listening, armed, heard, start, stop, arm };
}
