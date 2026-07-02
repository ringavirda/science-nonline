/**
 * MonitorScreen -- the live telemetry dashboard (the app's first tab).
 *
 * Reads the shared BLE stream (passed in from App so both tabs share one
 * connection) and renders the fix banner, staleness clock, GPS/IMU tiles, the
 * on-MCU LSI section and the SD REC / NO SD chip. Styling is NativeWind
 * (className); the palette lives in tailwind.config.js.
 */
import { ScrollView, Text, TouchableOpacity, View } from "react-native";

import { accelMagnitude, metresBetween, Telemetry } from "./telemetry";
import type { BleTelemetry } from "./useBleTelemetry";

type Tone = "ok" | "warn" | "bad" | "info" | "dim";

const TONE_TEXT: Record<Tone, string> = {
  ok: "text-ok",
  warn: "text-warn",
  bad: "text-bad",
  info: "text-info",
  dim: "text-dim",
};
const TONE_BORDER: Record<Tone, string> = {
  ok: "border-ok",
  warn: "border-warn",
  bad: "border-bad",
  info: "border-info",
  dim: "border-line",
};

const fmt = (v: number | undefined, d = 1): string =>
  v == null || !Number.isFinite(v) ? "--" : v.toFixed(d);

interface Banner {
  tone: Tone;
  title: string;
  sub: string;
}

function bannerFor(
  state: string,
  t: Telemetry | null,
  ageMs: number | null,
  error: string | null,
): Banner {
  if (state === "connected" && ageMs != null && ageMs > 3000) {
    return {
      tone: "bad",
      title: "NO DATA",
      sub: `silent for ${(ageMs / 1000).toFixed(0)} s -- board stalled or out of range`,
    };
  }
  if (state === "connected" && t) {
    if (t.fix === 1 && t.sats > 0) {
      return { tone: "ok", title: "GPS FIX", sub: `${t.sats} satellites locked` };
    }
    return {
      tone: "warn",
      title: "ACQUIRING FIX",
      sub: `${t.sats} satellites -- move to open sky`,
    };
  }
  if (state === "scanning")
    return { tone: "info", title: "SCANNING", sub: "looking for dtfit-gps..." };
  if (state === "connecting")
    return { tone: "info", title: "CONNECTING", sub: "negotiating link..." };
  if (state === "error")
    return { tone: "bad", title: "ERROR", sub: error ?? "unknown error" };
  return { tone: "dim", title: "NOT CONNECTED", sub: "tap CONNECT to start" };
}

function Tile({
  label,
  value,
  unit,
  toneClass = "text-ink",
}: {
  label: string;
  value: string;
  unit?: string;
  toneClass?: string;
}) {
  return (
    <View className="bg-card border border-line rounded-xl py-3 px-3.5 basis-[31%] grow">
      <Text className="text-dim text-[11px] font-semibold tracking-wider">
        {label}
      </Text>
      <View className="flex-row items-end mt-1.5">
        <Text className={`text-xl font-bold ${toneClass}`}>{value}</Text>
        {unit ? <Text className="text-dim text-xs ml-1 mb-0.5">{unit}</Text> : null}
      </View>
    </View>
  );
}

function Sparkline({
  values,
  barClass,
}: {
  values: number[];
  barClass: string;
}) {
  const clean = values.map((v) => (Number.isFinite(v) ? v : 0));
  const max = Math.max(1, ...clean);
  return (
    <View className="flex-row items-end h-10">
      {clean.map((v, i) => (
        <View
          key={i}
          className={`flex-1 rounded-[1px] ${barClass}`}
          style={{
            marginHorizontal: 0.5,
            height: `${Math.max(3, (v / max) * 100)}%` as const,
            opacity: 0.4 + 0.6 * (i / Math.max(1, clean.length - 1)),
          }}
        />
      ))}
    </View>
  );
}

function Section({ title }: { title: string }) {
  return (
    <Text className="text-dim text-xs font-bold tracking-widest mt-3.5 mb-2">
      {title}
    </Text>
  );
}

export default function MonitorScreen({
  ble,
  now,
}: {
  ble: BleTelemetry;
  now: number;
}) {
  const t = ble.telemetry;
  const ageMs = ble.lastRxAt != null ? now - ble.lastRxAt : null;
  const live = ble.state === "connected" && ageMs != null && ageMs <= 3000;
  const banner = bannerFor(ble.state, t, ageMs, ble.error);

  const connecting = ble.state === "scanning" || ble.state === "connecting";
  const active = connecting || ble.state === "connected";

  const speeds = ble.history.map((h) => h.spdKmph);
  const satsHist = ble.history.map((h) => h.sats);

  const estErr =
    t && t.schema === "lsi" && t.estLat != null && t.estLon != null
      ? metresBetween(t.lat, t.lon, t.estLat, t.estLon)
      : null;

  const pillBorder = live ? "border-ok" : active ? "border-info" : "border-line";
  const dotBg = live ? "bg-ok" : active ? "bg-info" : "bg-dim";

  return (
    <ScrollView contentContainerClassName="p-4">
      <View className="flex-row justify-between items-center mb-4">
        <View>
          <Text className="text-ink text-2xl font-bold">dtfit monitor</Text>
          <Text className="text-dim text-[13px] mt-0.5">realtime_gps_hw rig</Text>
        </View>
        <View className="items-end">
          <View
            className={`flex-row items-center border rounded-full px-2.5 py-1.5 ${pillBorder}`}
          >
            <View className={`w-2 h-2 rounded-full mr-1.5 ${dotBg}`} />
            <Text className="text-ink text-xs font-semibold">
              {ble.deviceName ?? (active ? "linking" : "offline")}
            </Text>
          </View>
          {t?.sd != null ? (
            // Tap to toggle recording: sd 0 = paused (OFF), 1 = recording (REC), 2 = NO SD.
            // Pause it when parked / walking in & out of the house; enable it for the drive.
            <TouchableOpacity
              activeOpacity={0.7}
              onPress={() => ble.setRecording(t.sd === 0)}
              className={`flex-row items-center border rounded-full px-2.5 py-1 mt-1.5 ${
                t.sd === 1 ? "border-ok" : t.sd === 2 ? "border-bad" : "border-line"
              }`}
            >
              <View
                className={`w-2 h-2 rounded-full mr-1.5 ${
                  t.sd === 1 ? "bg-ok" : t.sd === 2 ? "bg-bad" : "bg-dim"
                }`}
              />
              <Text
                className={`text-xs font-bold ${
                  t.sd === 1 ? "text-ok" : t.sd === 2 ? "text-bad" : "text-dim"
                }`}
              >
                {t.sd === 1 ? "REC" : t.sd === 2 ? "NO SD" : "OFF"}
              </Text>
            </TouchableOpacity>
          ) : null}
        </View>
      </View>

      {!ble.bluetoothOn ? (
        <View className="border border-bad rounded-2xl p-4 mb-3 bg-card">
          <Text className="text-bad text-[22px] font-extrabold tracking-widest">
            BLUETOOTH OFF
          </Text>
          <Text className="text-dim text-[13px] mt-1">
            enable Bluetooth to connect
          </Text>
        </View>
      ) : (
        <View
          className={`border rounded-2xl p-4 mb-3 bg-card ${TONE_BORDER[banner.tone]}`}
        >
          <Text
            className={`text-[22px] font-extrabold tracking-widest ${TONE_TEXT[banner.tone]}`}
          >
            {banner.title}
          </Text>
          <Text className="text-dim text-[13px] mt-1">{banner.sub}</Text>
        </View>
      )}

      <View className="flex-row justify-between mb-2">
        <Text className="text-dim text-xs">
          {ble.packetCount > 0 ? `#${ble.packetCount}` : "--"} packets
        </Text>
        <Text className="text-dim text-xs">
          {ageMs != null ? `${(ageMs / 1000).toFixed(1)} s ago` : "no data"}
        </Text>
        <Text className="text-dim text-xs">{t ? t.schema : "--"} schema</Text>
      </View>

      <Section title="POSITION" />
      <View className="flex-row flex-wrap gap-2">
        <Tile label="SPEED" value={fmt(t?.spdKmph)} unit="km/h" />
        <Tile
          label="SATS"
          value={t ? String(t.sats) : "--"}
          toneClass={t && t.sats >= 6 ? "text-ok" : t ? "text-warn" : "text-ink"}
        />
        <Tile
          label="HDOP"
          value={fmt(t?.hdop)}
          toneClass={t && t.hdop > 0 && t.hdop < 2 ? "text-ok" : "text-ink"}
        />
        <Tile label="ALT" value={fmt(t?.altM)} unit="m" />
        <Tile label="LAT" value={fmt(t?.lat, 6)} />
        <Tile label="LON" value={fmt(t?.lon, 6)} />
      </View>

      {ble.history.length > 1 ? (
        <View className="bg-card border border-line rounded-xl p-3 mt-2.5">
          <Text className="text-dim text-[11px] mb-1">speed (km/h)</Text>
          <Sparkline values={speeds} barClass="bg-accent" />
          <Text className="text-dim text-[11px] mb-1 mt-2.5">satellites</Text>
          <Sparkline values={satsHist} barClass="bg-ok" />
        </View>
      ) : null}

      <Section title="INERTIAL (IMU)" />
      <View className="flex-row flex-wrap gap-2">
        <Tile label="ACC X" value={fmt(t?.ax, 3)} unit="g" />
        <Tile label="ACC Y" value={fmt(t?.ay, 3)} unit="g" />
        <Tile label="ACC Z" value={fmt(t?.az, 3)} unit="g" />
        <Tile label="GYR X" value={fmt(t?.gx, 2)} unit="dps" />
        <Tile label="GYR Y" value={fmt(t?.gy, 2)} unit="dps" />
        <Tile label="GYR Z" value={fmt(t?.gz, 2)} unit="dps" />
      </View>
      {t ? (
        <Text className="text-dim text-xs mt-2 italic">
          |a| = {fmt(accelMagnitude(t), 3)} g (near 1.000 g at rest confirms the
          IMU is live)
        </Text>
      ) : null}

      {t?.schema === "lsi" ? (
        <>
          <Section title="ON-MCU FILTER (LSI)" />
          <View className="flex-row flex-wrap gap-2">
            <Tile
              label="MODE"
              value={t.mode === 1 ? "MOVING" : "STILL"}
              toneClass={t.mode === 1 ? "text-accent" : "text-dim"}
            />
            <Tile label="COST" value={fmt(t.costUs, 0)} unit="us" />
            <Tile
              label="EST ERR"
              value={estErr != null ? fmt(estErr, 1) : "--"}
              unit="m"
              toneClass={estErr != null && estErr < 5 ? "text-ok" : "text-ink"}
            />
          </View>
        </>
      ) : null}

      <View className="h-4" />
      <TouchableOpacity
        activeOpacity={0.85}
        onPress={active ? ble.disconnect : ble.connect}
        className={`border rounded-2xl py-4 items-center ${
          active ? "bg-card2 border-line" : "bg-accent border-accent"
        }`}
      >
        <Text
          className={`text-base font-extrabold tracking-widest ${
            active ? "text-ink" : "text-bg"
          }`}
        >
          {ble.state === "error" ? "RETRY" : active ? "DISCONNECT" : "CONNECT"}
        </Text>
      </TouchableOpacity>
      <View className="h-6" />
    </ScrollView>
  );
}
