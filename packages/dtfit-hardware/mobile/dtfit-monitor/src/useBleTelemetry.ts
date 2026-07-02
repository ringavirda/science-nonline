/**
 * useBleTelemetry -- connect to the dtfit-gps rig and stream its telemetry.
 *
 * Mirrors backend.py's bleak reader: scan for the device, subscribe to the one
 * notify characteristic, decode each base64 chunk to a CSV line and parse it.
 * The identifiers below MUST match firmware/nano_ble_telemetry & nano_lsi_log.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { PermissionsAndroid, Platform } from "react-native";
import {
  BleError,
  BleErrorCode,
  BleManager,
  Characteristic,
  Device,
  State,
  Subscription,
} from "react-native-ble-plx";
import { Buffer } from "buffer";

import { parseTelemetry, Telemetry } from "./telemetry";

export const DEVICE_NAME = "dtfit-gps";
export const SERVICE_UUID = "9a1e0000-1b2c-4f3a-8d5e-6f7a8b9c0d10";
export const TELE_UUID = "9a1e0001-1b2c-4f3a-8d5e-6f7a8b9c0d10";
export const CTRL_UUID = "9a1e0002-1b2c-4f3a-8d5e-6f7a8b9c0d10"; // write "1"/"0" to record/pause

export type ConnState =
  | "idle"
  | "scanning"
  | "connecting"
  | "connected"
  | "error";

const HISTORY = 48; // samples kept for the sparklines (~48 s at 1 Hz)

// One BleManager per app process. Created lazily so importing the module (e.g.
// during a Fast Refresh) never touches native code before the tree mounts.
let manager: BleManager | null = null;
function getManager(): BleManager {
  if (manager === null) manager = new BleManager();
  return manager;
}

async function requestPermissions(): Promise<boolean> {
  if (Platform.OS !== "android") return true;
  const api =
    typeof Platform.Version === "number"
      ? Platform.Version
      : parseInt(String(Platform.Version), 10);
  if (api >= 31) {
    // Android 12+: runtime BLUETOOTH_SCAN + BLUETOOTH_CONNECT.
    const res = await PermissionsAndroid.requestMultiple([
      PermissionsAndroid.PERMISSIONS.BLUETOOTH_SCAN,
      PermissionsAndroid.PERMISSIONS.BLUETOOTH_CONNECT,
    ]);
    return (
      res[PermissionsAndroid.PERMISSIONS.BLUETOOTH_SCAN] ===
        PermissionsAndroid.RESULTS.GRANTED &&
      res[PermissionsAndroid.PERMISSIONS.BLUETOOTH_CONNECT] ===
        PermissionsAndroid.RESULTS.GRANTED
    );
  }
  // Android 11 and below: BLE scanning is gated on fine location.
  const res = await PermissionsAndroid.request(
    PermissionsAndroid.PERMISSIONS.ACCESS_FINE_LOCATION,
  );
  return res === PermissionsAndroid.RESULTS.GRANTED;
}

export interface BleTelemetry {
  state: ConnState;
  error: string | null;
  bluetoothOn: boolean;
  deviceName: string | null;
  telemetry: Telemetry | null;
  history: Telemetry[];
  lastRxAt: number | null;
  packetCount: number;
  connect: () => void;
  disconnect: () => void;
  setRecording: (on: boolean) => void; // write the record/pause command to the board
}

export function useBleTelemetry(): BleTelemetry {
  const [state, setState] = useState<ConnState>("idle");
  const [error, setError] = useState<string | null>(null);
  const [bluetoothOn, setBluetoothOn] = useState(true);
  const [deviceName, setDeviceName] = useState<string | null>(null);
  const [telemetry, setTelemetry] = useState<Telemetry | null>(null);
  const [history, setHistory] = useState<Telemetry[]>([]);
  const [lastRxAt, setLastRxAt] = useState<number | null>(null);
  const [packetCount, setPacketCount] = useState(0);

  const deviceRef = useRef<Device | null>(null);
  const monitorSub = useRef<Subscription | null>(null);
  const disconnSub = useRef<Subscription | null>(null);

  // Track the Bluetooth adapter power state for the "turn BT on" banner.
  useEffect(() => {
    const sub = getManager().onStateChange(
      (s) => setBluetoothOn(s === State.PoweredOn),
      true,
    );
    return () => sub.remove();
  }, []);

  const cleanupConnection = useCallback(() => {
    monitorSub.current?.remove();
    monitorSub.current = null;
    disconnSub.current?.remove();
    disconnSub.current = null;
  }, []);

  const handleChunk = useCallback((text: string) => {
    let got = false;
    for (const ln of text.split(/[\r\n]+/)) {
      const t = parseTelemetry(ln);
      if (t) {
        got = true;
        setTelemetry(t);
        setHistory((h) => {
          const next = h.length >= HISTORY ? h.slice(1) : h.slice();
          next.push(t);
          return next;
        });
      }
    }
    if (got) {
      setLastRxAt(Date.now());
      setPacketCount((c) => c + 1);
      setState("connected");
    }
  }, []);

  const onFound = useCallback(
    async (found: Device) => {
      const mgr = getManager();
      try {
        setState("connecting");
        const dev = await mgr.connectToDevice(found.id, { timeout: 12000 });
        // Enlarge the MTU so the full CSV line (~150 B for the lsi schema) fits
        // in a single notification; the default 23 B MTU would truncate it.
        await dev.requestMTU(512).catch(() => undefined);
        await dev.discoverAllServicesAndCharacteristics();
        deviceRef.current = dev;
        setDeviceName(dev.name ?? found.name ?? DEVICE_NAME);

        disconnSub.current = mgr.onDeviceDisconnected(dev.id, () => {
          cleanupConnection();
          deviceRef.current = null;
          setState("idle");
        });

        monitorSub.current = dev.monitorCharacteristicForService(
          SERVICE_UUID,
          TELE_UUID,
          (err: BleError | null, ch: Characteristic | null) => {
            if (err) {
              // Cancellation/disconnect errors are expected on teardown.
              if (
                err.errorCode !== BleErrorCode.OperationCancelled &&
                err.errorCode !== BleErrorCode.DeviceDisconnected
              ) {
                setError(err.message);
              }
              return;
            }
            if (ch?.value) {
              handleChunk(Buffer.from(ch.value, "base64").toString("utf8"));
            }
          },
        );
        setState("connected");
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
        setState("error");
      }
    },
    [cleanupConnection, handleChunk],
  );

  const connect = useCallback(() => {
    void (async () => {
      setError(null);
      setPacketCount(0);
      setHistory([]);
      setTelemetry(null);
      setLastRxAt(null);

      const granted = await requestPermissions();
      if (!granted) {
        setError("Bluetooth permission denied");
        setState("error");
        return;
      }
      const mgr = getManager();
      if ((await mgr.state()) !== State.PoweredOn) {
        setError("Turn Bluetooth on and retry");
        setState("error");
        return;
      }

      setState("scanning");
      mgr.startDeviceScan(null, { allowDuplicates: false }, (err, device) => {
        if (err) {
          setError(err.message);
          setState("error");
          mgr.stopDeviceScan();
          return;
        }
        if (!device) return;
        const nm = device.name ?? device.localName ?? "";
        const svcMatch =
          device.serviceUUIDs?.some((u) => u.toLowerCase() === SERVICE_UUID) ??
          false;
        if (nm === DEVICE_NAME || svcMatch) {
          mgr.stopDeviceScan();
          void onFound(device);
        }
      });
    })();
  }, [onFound]);

  // Toggle SD recording on the board (telemetry keeps streaming either way). Write-with-response
  // so a failed toggle surfaces; the board echoes the new state back in the `sd` column.
  const setRecording = useCallback((on: boolean) => {
    const dev = deviceRef.current;
    if (!dev) return;
    const b64 = Buffer.from(on ? "1" : "0", "utf8").toString("base64");
    dev
      .writeCharacteristicWithResponseForService(SERVICE_UUID, CTRL_UUID, b64)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  const disconnect = useCallback(() => {
    void (async () => {
      const mgr = getManager();
      mgr.stopDeviceScan();
      cleanupConnection();
      const dev = deviceRef.current;
      deviceRef.current = null;
      if (dev) {
        try {
          await mgr.cancelDeviceConnection(dev.id);
        } catch {
          // already disconnected
        }
      }
      setState("idle");
    })();
  }, [cleanupConnection]);

  // Tear the link down if the screen unmounts.
  useEffect(
    () => () => {
      const mgr = getManager();
      mgr.stopDeviceScan();
      cleanupConnection();
      const dev = deviceRef.current;
      if (dev) mgr.cancelDeviceConnection(dev.id).catch(() => undefined);
    },
    [cleanupConnection],
  );

  return {
    state,
    error,
    bluetoothOn,
    deviceName,
    telemetry,
    history,
    lastRxAt,
    packetCount,
    connect,
    disconnect,
    setRecording,
  };
}
