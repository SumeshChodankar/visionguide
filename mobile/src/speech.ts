/**
 * Priority speech queue on top of expo-speech (on-device TTS).
 *
 *  - CRITICAL / interrupt utterances cut off whatever is playing.
 *  - Queue is ordered by priority; low-priority items older than 8 s are dropped
 *    (a stale "door on your right" is worse than silence).
 *  - Haptics fire with the utterance so deaf-blind users get a signal too.
 *  - When a screen reader is running, text is also announced through it so
 *    VoiceOver/TalkBack users keep their own voice settings.
 *  - Answers to the user's own instructions are spoken word for word (never
 *    dropped as stale) and then repeated every few seconds until the next
 *    instruction or Stop, so nothing is missed in a noisy street.
 */
import * as Haptics from 'expo-haptics';
import * as Speech from 'expo-speech';
import { AccessibilityInfo } from 'react-native';

import { CONFIG } from './config';
import { Priority, type Utterance } from './types';

type Item = Utterance & { at: number; keep?: boolean };
type Listener = (u: Utterance) => void;
type RepeatListener = (text: string | null) => void;

class SpeechQueue {
  private queue: Item[] = [];
  private speaking = false;
  private muted = false;
  private listeners = new Set<Listener>();
  private repeatListeners = new Set<RepeatListener>();
  private screenReader = false;
  private repeatText: string | null = null;
  private repeatTimer: ReturnType<typeof setTimeout> | null = null;
  private repeatPaused = false;

  constructor() {
    AccessibilityInfo.isScreenReaderEnabled().then((v) => (this.screenReader = v));
    AccessibilityInfo.addEventListener('screenReaderChanged', (v) => (this.screenReader = v));
  }

  onSpeak(fn: Listener) {
    this.listeners.add(fn);
    return () => {
      this.listeners.delete(fn);
    };
  }

  onRepeatChange(fn: RepeatListener) {
    this.repeatListeners.add(fn);
    return () => {
      this.repeatListeners.delete(fn);
    };
  }

  /**
   * Queue utterances. With `answer: true` (a reply to the user's own instruction) they are
   * kept until spoken and become the text that repeats until the next instruction.
   */
  say(utts: Utterance[] | undefined, opts: { answer?: boolean } = {}) {
    if (!utts?.length) return;
    const now = Date.now();
    if (opts.answer) this.setRepeat(utts.map((u) => u.text.trim()).filter(Boolean).join(' '));
    else this.cancelRepeatTimer(); // fresh live info goes first; the repeat resumes after it
    for (const u of utts) {
      const item: Item = { ...u, at: now, keep: opts.answer };
      if (u.interrupt || u.priority >= Priority.CRITICAL) {
        this.queue = this.queue.filter((q) => q.keep || q.priority >= Priority.WARNING);
        this.queue.unshift(item);
        this.interrupt();
      } else {
        this.queue.push(item);
      }
    }
    this.queue.sort((a, b) => b.priority - a.priority || a.at - b.at);
    this.pump();
  }

  /**
   * While walking, live guidance (turns, hazards) must not compete with the repeated answer:
   * repeating pauses while you walk and resumes when you stop.
   */
  setRepeatPaused(paused: boolean) {
    if (paused === this.repeatPaused) return;
    this.repeatPaused = paused;
    if (paused) this.cancelRepeatTimer();
    else if (!this.speaking && !this.queue.length) this.scheduleRepeat();
  }

  /** A new instruction is coming: stop talking and stop repeating the last answer. */
  clearRepeat() {
    this.cancelRepeatTimer();
    if (this.repeatText !== null) {
      this.repeatText = null;
      this.repeatListeners.forEach((l) => l(null));
    }
  }

  private setRepeat(text: string) {
    this.cancelRepeatTimer();
    this.repeatText = text || null;
    this.repeatListeners.forEach((l) => l(this.repeatText));
  }

  private cancelRepeatTimer() {
    if (this.repeatTimer) clearTimeout(this.repeatTimer);
    this.repeatTimer = null;
  }

  private scheduleRepeat() {
    if (!this.repeatText || this.repeatTimer || this.muted || this.repeatPaused) return;
    this.repeatTimer = setTimeout(() => {
      this.repeatTimer = null;
      if (!this.repeatText || this.speaking || this.muted || this.repeatPaused || this.queue.length) return;
      this.queue.push({
        text: this.repeatText, priority: Priority.GUIDANCE, category: 'answer', interrupt: false,
        haptic: 'none', at: Date.now(), keep: true,
      });
      this.pump();
    }, CONFIG.repeatGapMs);
  }

  /** Speak a local, client-generated message (errors, permissions). */
  local(text: string, priority: Priority = Priority.GUIDANCE) {
    this.say([{ text, priority, category: 'system', interrupt: false, haptic: 'none' }]);
  }

  async stopAll() {
    this.clearRepeat();
    this.queue = [];
    this.speaking = false;
    await Speech.stop();
  }

  /** Silence while the microphone records, so TTS isn't transcribed. */
  async hold() {
    this.clearRepeat();
    this.muted = true;
    await Speech.stop();
    this.speaking = false;
  }

  release() {
    this.muted = false;
    this.pump();
  }

  private interrupt() {
    if (this.speaking) {
      this.speaking = false;
      Speech.stop();
    }
  }

  private pump() {
    if (this.speaking || this.muted) return;
    const now = Date.now();
    this.queue = this.queue.filter((q) => q.keep || q.priority >= Priority.WARNING || now - q.at < 8000);
    const next = this.queue.shift();
    if (!next) {
      this.scheduleRepeat();
      return;
    }
    this.speaking = true;
    this.haptic(next.haptic);
    this.listeners.forEach((l) => l(next));
    const done = () => {
      this.speaking = false;
      this.pump();
    };
    if (this.screenReader) {
      // Route through VoiceOver/TalkBack (user's own voice + rate). No completion
      // callback exists, so estimate duration (~14 chars/s) before the next item.
      AccessibilityInfo.announceForAccessibility(next.text);
      setTimeout(done, Math.max(1200, next.text.length * 70));
      return;
    }
    Speech.speak(next.text, {
      rate: CONFIG.speechRate,
      language: 'en-US',
      onDone: done,
      onStopped: () => {
        this.speaking = false;
      },
      onError: done,
    });
  }

  private haptic(kind: Utterance['haptic']) {
    if (kind === 'warning') Haptics.notificationAsync(Haptics.NotificationFeedbackType.Warning);
    else if (kind === 'success') Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
    else if (kind === 'light') Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
  }
}

export const speech = new SpeechQueue();
