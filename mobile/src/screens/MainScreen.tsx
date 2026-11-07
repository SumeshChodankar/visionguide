import { CameraView, useCameraPermissions } from 'expo-camera';
import { useKeepAwake } from 'expo-keep-awake';
import { useCallback, useEffect, useRef, useState } from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { api } from '../api';
import { listeningCue } from '../beep';
import { API_URL, BUILD_TAG, CONFIG } from '../config';
import { useCourseGuide } from '../hooks/useCourseGuide';
import { useLocationTracking } from '../hooks/useLocationTracking';
import { useSceneLoop } from '../hooks/useSceneLoop';
import { useSensors } from '../hooks/useSensors';
import { useVoiceCommand } from '../hooks/useVoiceCommand';
import { useWakeWord } from '../hooks/useWakeWord';
import { speech } from '../speech';
import { Priority, type AgentResponse, type Phase, type Utterance } from '../types';

const PHASE_LABEL: Record<Phase, string> = {
  idle: 'Ready',
  outdoor: 'Walking to building',
  approach: 'Finding entrance',
  indoor: 'Finding room',
  arrived: 'Arrived',
};

export default function MainScreen() {
  useKeepAwake(); // screen must stay on while guiding
  const camera = useRef<CameraView>(null);
  const [camPerm, requestCamPerm] = useCameraPermissions();
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [phase, setPhase] = useState<Phase>('idle');
  const [goal, setGoal] = useState<string | null>(null);
  const [caption, setCaption] = useState<Utterance | null>(null);
  const [liveOverride, setLiveOverride] = useState<boolean | null>(null);
  const [online, setOnline] = useState(true);
  const [trace, setTrace] = useState<string | null>(null);
  const [youSaid, setYouSaid] = useState<string | null>(null);
  const [repeating, setRepeating] = useState(false);
  const [thinking, setThinking] = useState(false);
  const busyRef = useRef(false); // a question is in flight
  const [stepIndex, setStepIndex] = useState<number | null>(null);
  const lookNowRef = useRef<() => void>(() => {});

  // ---- phone sensors: compass, steps, gyroscope, light, barometer --------
  const { sensors, summary: sensorSummary, takeFloorChange } = useSensors({
    onTurn: () => lookNowRef.current(), // you turned: the camera looks again right away
    onFloorChange: (n) =>
      speech.local(`${n > 0 ? 'Up' : 'Down'} ${Math.abs(n) === 1 ? 'one floor' : `${Math.abs(n)} floors`}.`, Priority.GUIDANCE),
    onDark: () => speech.local("It's very dark here, so the camera can't see well. Rely on your cane or guide dog.", Priority.WARNING),
  });
  useEffect(() => speech.setRepeatPaused(sensorSummary.walking), [sensorSummary.walking]);

  // ---- session bootstrap -------------------------------------------------
  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const s = await api.createSession();
        if (!alive) return;
        setSessionId(s.session_id);
        speech.say(s.utterances);
      } catch {
        setOnline(false);
        speech.local(`I can't reach the VisionGuide server at ${API_URL}. Check that the phone and laptop are on the same Wi-Fi.`);
      }
    })();
    return () => {
      alive = false;
    };
  }, []);

  useEffect(() => {
    if (camPerm && !camPerm.granted && camPerm.canAskAgain) requestCamPerm();
  }, [camPerm, requestCamPerm]);

  useEffect(() => speech.onSpeak(setCaption), []);
  useEffect(() => speech.onRepeatChange((t) => setRepeating(!!t)), []);

  const live = liveOverride ?? (phase !== 'idle' && phase !== 'arrived');

  // ---- one handler for every agent response -----------------------------
  // `answer` = reply to the user's own instruction: spoken word for word, then repeated until the next one.
  const sceneRef = useRef<{ captureNow: (m: 'auto' | 'describe' | 'read') => Promise<void> } | null>(null);
  const handle = useCallback((r: AgentResponse, answer = false) => {
    setOnline(true);
    speech.say(r.utterances, { answer });
    if (r.handled_by) {
      const tools = (r.tool_trace ?? []).filter((t) => t.tool).map((t) => t.tool);
      setTrace(r.handled_by === 'brain' ? `🧠 ${tools.length ? tools.join(' → ') : 'answered directly'}` : '⚡ rules');
    }
    setPhase(r.phase);
    setStepIndex(r.step_index ?? null);
    const d = r.destination;
    setGoal(d ? (d.room ? `Room ${d.room}` : d.sign_text || d.place_name || d.target_kind || d.raw) : null);
    if (r.client_action === 'capture_frame') {
      sceneRef.current?.captureNow(r.intent?.name === 'describe' ? 'describe' : 'auto');
    } else if (r.client_action === 'capture_frame_read') {
      sceneRef.current?.captureNow('read');
    }
  }, []);

  const background = useCallback((r: AgentResponse) => handle(r, false), [handle]);
  const { fix } = useLocationTracking(sessionId, background, sensors, takeFloorChange);
  const scene = useSceneLoop(sessionId, camera, fix, live && !!camPerm?.granted, background);
  sceneRef.current = scene;
  lookNowRef.current = scene.lookNow;
  const course = useCourseGuide(sessionId, phase, stepIndex, fix, sensors);
  const voice = useVoiceCommand(sessionId, (r) => handle(r, true), scene.snapshot);

  /** Every instruction (wake word, button, tap) goes through here. */
  const runCommand = useCallback(
    async (text: string, withFrame = true) => {
      if (!sessionId) return;
      const isStop = /^\s*(stop|cancel|quiet|be quiet|silence|pause|shut up)\b/i.test(text);
      // One question at a time: extra presses used to start parallel requests that slowed every answer.
      // "Stop" always goes through.
      if (busyRef.current && !isStop) {
        speech.local('Still working on your last question.', Priority.GUIDANCE);
        return;
      }
      await speech.stopAll(); // a new instruction ends the previous answer and its repeat
      setYouSaid(text);
      if (isStop) setLiveOverride(null);
      busyRef.current = true;
      setThinking(true);
      try {
        if (!isStop) speech.local('One moment.', 0);
        const frame = withFrame && !isStop && camPerm?.granted ? await scene.snapshot() : null;
        let r;
        try {
          r = await api.ask(sessionId, isStop ? 'stop' : text, frame);
        } catch (e) {
          if (!frame) throw e;
          // The photo upload failed (seen on some Android phones): still answer the question without it.
          console.warn('[ask with photo failed, retrying without]', String(e));
          setTrace(`⚠ photo upload failed: ${String(e).slice(0, 80)}`);
          r = await api.ask(sessionId, text, null);
        }
        // "stop" is answered but not repeated: repeating "navigation stopped" would defeat the point.
        handle(r, !isStop);
      } catch (e) {
        console.warn('[command failed]', text, String(e));
        setTrace(`⚠ ${String(e).slice(0, 120)}`);
        speech.local('The server did not respond. Trying again may help.');
      } finally {
        busyRef.current = false;
        setThinking(false);
      }
    },
    [sessionId, camPerm?.granted, scene, handle],
  );

  const wake = useWakeWord({
    onWake: () => {
      speech.stopAll();
      listeningCue();
    },
    onCommand: (text) => runCommand(text),
    onError: (msg) => speech.local(msg, Priority.WARNING),
  });

  const quick = (text: string, withFrame = false) => () => runCommand(text, withFrame);

  // Web: the big button switches on hands-free mode (this first tap also lets the browser speak),
  // then works as "listen now". Phones (Expo Go): tap-to-talk recording as before.
  const handsFree = wake.supported;
  const onTalk = () => {
    if (!handsFree) return voice.toggle();
    if (!wake.listening) {
      if (wake.start()) {
        speech.local(`Hands-free is on. Say “${CONFIG.wakeWord}”, then your instruction. For example: ${CONFIG.wakeWord}, where am I?`);
      } else {
        speech.local('Voice control is not available in this browser. Use Chrome or Edge.');
      }
      return;
    }
    speech.stopAll();
    listeningCue();
    wake.arm();
  };

  const talkLabel = handsFree
    ? !wake.listening
      ? 'Tap to start hands-free'
      : wake.armed
        ? 'Listening… say your instruction'
        : thinking
          ? 'Thinking…'
          : `Say “${CONFIG.wakeWord}”, or tap and speak`
    : voice.recording ? 'Listening. Tap to send.' : voice.busy || thinking ? 'Thinking…' : 'Tap to speak';

  return (
    <SafeAreaView style={styles.root} edges={['top', 'bottom']}>
      {/* Camera + caption: the caption helps low-vision users and sighted judges */}
      <View style={styles.cameraWrap} accessible={false} importantForAccessibility="no-hide-descendants">
        {camPerm?.granted ? (
          <CameraView ref={camera} style={StyleSheet.absoluteFill} facing="back" animateShutter={false} />
        ) : (
          <Text style={styles.camMissing}>Camera permission needed</Text>
        )}
        <View style={styles.badgeRow}>
          <Text style={[styles.badge, !online && styles.badgeOffline]}>
            {online ? PHASE_LABEL[phase] : 'Offline'}
          </Text>
          {live && <Text style={[styles.badge, styles.badgeLive]}>LIVE</Text>}
          {wake.listening && <Text style={[styles.badge, styles.badgeMic]}>🎙 {CONFIG.wakeWord}</Text>}
          {scene.fastLatency != null && <Text style={styles.badge}>⚡{(scene.fastLatency / 1000).toFixed(1)}s</Text>}
          {scene.lastLatency != null && <Text style={styles.badge}>👁{(scene.lastLatency / 1000).toFixed(1)}s</Text>}
        </View>
        <Text style={styles.sensorLine} importantForAccessibility="no" accessibilityElementsHidden>
          {[
            sensorSummary.heading != null ? `🧭 ${sensorSummary.heading}°` : '🧭 –',
            sensorSummary.walking ? `🚶 ${sensorSummary.steps}` : '🧍',
            sensorSummary.lux != null ? `💡${sensorSummary.lux}` : null,
            sensorSummary.baro ? '⛰ baro' : null,
            course.status || null,
            BUILD_TAG,
          ].filter(Boolean).join('  ·  ')}
        </Text>
      </View>

      <View style={styles.captionBox} accessibilityLiveRegion="polite">
        {goal && <Text style={styles.goal}>Goal: {goal}</Text>}
        {(wake.heard || youSaid) && (
          <Text style={styles.you} numberOfLines={2}>
            You: “{wake.heard ?? youSaid}”
          </Text>
        )}
        {trace && (
          <Text style={styles.trace} importantForAccessibility="no" accessibilityElementsHidden>
            {trace}
          </Text>
        )}
        <Text
          style={[
            styles.caption,
            caption?.priority === Priority.CRITICAL && styles.captionCritical,
            caption?.category === 'arrival' && styles.captionArrived,
          ]}
          numberOfLines={8}
        >
          {caption?.text ?? `Say “${CONFIG.wakeWord}, take me to Room 443B”`}
        </Text>
        {repeating && (
          <Text style={styles.trace} importantForAccessibility="no" accessibilityElementsHidden>
            🔁 Repeating until your next instruction
          </Text>
        )}
      </View>

      {/* The main control: huge, centred, magic-tap enabled */}
      <Pressable
        onPress={onTalk}
        onMagicTap={onTalk}
        disabled={!sessionId || voice.busy}
        style={({ pressed }) => [
          styles.talk,
          (voice.recording || wake.armed) && styles.talkActive,
          pressed && styles.pressed,
        ]}
        accessibilityRole="button"
        accessibilityLabel={talkLabel}
        accessibilityHint="Speak a command such as: take me to room 443B, find the elevator, or where am I."
      >
        <Text style={styles.talkText}>{talkLabel}</Text>
      </Pressable>

      <View style={styles.row}>
        <Btn label="What's ahead" onPress={quick('What is in front of me?', true)} />
        <Btn label="Where am I" onPress={quick('Where am I?', true)} />
        <Btn label="Read sign" onPress={quick('Read the sign', true)} />
      </View>
      <View style={styles.row}>
        <Btn label="Repeat" onPress={quick('Repeat')} />
        <Btn
          label={live ? 'Live view on' : 'Live view off'}
          hint="Turns continuous camera guidance on or off"
          onPress={() => setLiveOverride(!live)}
        />
        <Btn label="Stop" danger onPress={quick('stop')} />
      </View>
    </SafeAreaView>
  );
}

function Btn({ label, onPress, hint, danger }: { label: string; onPress: () => void; hint?: string; danger?: boolean }) {
  return (
    <Pressable
      onPress={onPress}
      style={({ pressed }) => [styles.btn, danger && styles.btnDanger, pressed && styles.pressed]}
      accessibilityRole="button"
      accessibilityLabel={label}
      accessibilityHint={hint}
    >
      <Text style={styles.btnText}>{label}</Text>
    </Pressable>
  );
}

// High contrast (yellow on black ≈ 19:1), large type, 64 pt+ targets.
const YELLOW = '#FFD60A';
const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: '#000' },
  cameraWrap: { flex: 3, backgroundColor: '#111', overflow: 'hidden' },
  camMissing: { color: '#fff', fontSize: 20, textAlign: 'center', marginTop: 80 },
  badgeRow: { position: 'absolute', top: 10, left: 10, flexDirection: 'row', gap: 8 },
  badge: {
    color: '#000', backgroundColor: YELLOW, fontWeight: '800', fontSize: 14,
    paddingHorizontal: 10, paddingVertical: 4, borderRadius: 8, overflow: 'hidden',
  },
  badgeLive: { backgroundColor: '#FF453A', color: '#fff' },
  badgeOffline: { backgroundColor: '#8E8E93', color: '#fff' },
  badgeMic: { backgroundColor: '#30D158', color: '#000' },
  sensorLine: {
    position: 'absolute', bottom: 8, left: 10, right: 10, color: '#fff', fontSize: 13, fontWeight: '700',
    backgroundColor: 'rgba(0,0,0,0.55)', paddingHorizontal: 8, paddingVertical: 4, borderRadius: 8, overflow: 'hidden',
  },
  you: { color: '#8E8E93', fontSize: 16, fontStyle: 'italic', marginBottom: 4 },
  captionBox: { paddingHorizontal: 16, paddingVertical: 12, minHeight: 120 },
  goal: { color: YELLOW, fontSize: 16, fontWeight: '700', marginBottom: 4 },
  trace: { color: '#8E8E93', fontSize: 13, marginBottom: 4 },
  caption: { color: '#fff', fontSize: 24, fontWeight: '700', lineHeight: 30 },
  captionCritical: { color: '#FF453A' },
  captionArrived: { color: '#30D158' },
  talk: {
    flex: 2, marginHorizontal: 16, marginBottom: 12, borderRadius: 28, backgroundColor: YELLOW,
    alignItems: 'center', justifyContent: 'center', minHeight: 140,
  },
  talkActive: { backgroundColor: '#FF453A' },
  talkText: { color: '#000', fontSize: 32, fontWeight: '900', textAlign: 'center', paddingHorizontal: 12 },
  row: { flexDirection: 'row', gap: 10, paddingHorizontal: 16, marginBottom: 10 },
  btn: {
    flex: 1, minHeight: 64, borderRadius: 16, borderWidth: 3, borderColor: YELLOW,
    alignItems: 'center', justifyContent: 'center', paddingHorizontal: 6,
  },
  btnDanger: { borderColor: '#FF453A' },
  btnText: { color: '#fff', fontSize: 17, fontWeight: '800', textAlign: 'center' },
  pressed: { opacity: 0.6 },
});
