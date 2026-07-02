/**
 * dtfit monitor -- a two-tab live view of the realtime_gps_hw rig over BLE.
 *
 *   MONITOR -- the telemetry dashboard (fix banner, GPS/IMU tiles, on-MCU LSI,
 *              and the SD REC / NO SD status).
 *   MAP     -- a live map of the reported lat/lon, so you can eyeball whether the
 *              rig's fix lands on the real road.
 *
 * The BLE hook lives here in the shell and is shared with both tabs (one
 * connection). Both screens stay mounted -- toggled with `display` -- so the map
 * keeps its tiles + trail and the link stays up when you switch tabs. Styling is
 * Tailwind via NativeWind (className); the palette lives in tailwind.config.js.
 */
import { useEffect, useState } from "react";
import { Text, TouchableOpacity, View } from "react-native";
import { StatusBar } from "expo-status-bar";
import { colorScheme } from "nativewind";
import {
  SafeAreaProvider,
  useSafeAreaInsets,
} from "react-native-safe-area-context";

import MapScreen from "./src/MapScreen";
import MonitorScreen from "./src/MonitorScreen";
import { useBleTelemetry } from "./src/useBleTelemetry";

type Tab = "monitor" | "map";

function TabButton({
  label,
  active,
  onPress,
}: {
  label: string;
  active: boolean;
  onPress: () => void;
}) {
  return (
    <TouchableOpacity
      onPress={onPress}
      activeOpacity={0.8}
      className="flex-1 items-center py-2.5"
    >
      <Text
        className={`text-sm font-bold tracking-wider ${
          active ? "text-accent" : "text-dim"
        }`}
      >
        {label}
      </Text>
      <View
        className={`h-0.5 w-10 mt-1.5 rounded-full ${
          active ? "bg-accent" : "bg-transparent"
        }`}
      />
    </TouchableOpacity>
  );
}

export default function App() {
  // SafeAreaProvider supplies the real device insets (status bar / nav bar) so the
  // tab bar clears the Android navigation buttons instead of hiding behind them.
  return (
    <SafeAreaProvider>
      <AppInner />
    </SafeAreaProvider>
  );
}

function AppInner() {
  const insets = useSafeAreaInsets();
  const ble = useBleTelemetry();
  const [now, setNow] = useState(Date.now());
  const [tab, setTab] = useState<Tab>("monitor");
  // Manual light/dark toggle (default dark). Driving into bright sun, dark tiles wash out;
  // tap the sun/moon to flip. colorScheme drives the CSS-variable palette (global.css).
  const [dark, setDark] = useState(true);
  useEffect(() => {
    colorScheme.set(dark ? "dark" : "light");
  }, [dark]);

  // Re-render twice a second so the "last packet" age and its colour update
  // even when no new packet has arrived.
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 500);
    return () => clearInterval(id);
  }, []);

  const t = ble.telemetry;
  const ageMs = ble.lastRxAt != null ? now - ble.lastRxAt : null;
  const live = ble.state === "connected" && ageMs != null && ageMs <= 3000;

  return (
    <View className="flex-1 bg-bg" style={{ paddingTop: insets.top }}>
      <StatusBar style={dark ? "light" : "dark"} />
      <View
        className="flex-1"
        style={{ display: tab === "monitor" ? "flex" : "none" }}
      >
        <MonitorScreen ble={ble} now={now} />
      </View>
      <View
        className="flex-1"
        style={{ display: tab === "map" ? "flex" : "none" }}
      >
        <MapScreen lat={t?.lat} lon={t?.lon} fix={t?.fix} live={live} dark={dark} />
      </View>
      {/* paddingBottom = nav-bar inset so the buttons sit above the system nav. */}
      <View
        className="flex-row border-t border-line bg-bg items-center"
        style={{ paddingBottom: insets.bottom }}
      >
        <TabButton
          label="MONITOR"
          active={tab === "monitor"}
          onPress={() => setTab("monitor")}
        />
        <TabButton
          label="MAP"
          active={tab === "map"}
          onPress={() => setTab("map")}
        />
        <TouchableOpacity
          onPress={() => setDark((d) => !d)}
          activeOpacity={0.7}
          className="px-4 py-2.5"
          accessibilityLabel="Toggle light or dark theme"
        >
          <Text className="text-lg">{dark ? "☀️" : "🌙"}</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}
