# dtfit monitor — phone-side live view of the rig (BLE)

A one-screen **React Native (Expo)** app that connects to the `realtime_gps_hw`
board over Bluetooth LE and shows its telemetry live, so you can tell at a glance
whether the board is working out in the field — GPS fix, satellite count, speed,
IMU, and (with the `nano_lsi_log` firmware) the on-MCU LSI estimate.

It is the phone-side twin of the rig's other two faces:

| Side | Lives in | Role |
|---|---|---|
| Board | [`../../src/dtfit_hardware/firmware/`](../../src/dtfit_hardware/firmware/) | emits the `dtfit-gps` GATT service |
| PC | [`../../src/dtfit_hardware/backend.py`](../../src/dtfit_hardware/backend.py) | `bleak` reader (`python backend.py ble`) |
| **Phone** | **this app** | **untethered live dashboard** |

It speaks the exact same GATT contract as the firmware and `backend.py`:

- **Device name:** `dtfit-gps`
- **Service UUID:** `9a1e0000-1b2c-4f3a-8d5e-6f7a8b9c0d10`
- **Notify characteristic:** `9a1e0001-1b2c-4f3a-8d5e-6f7a8b9c0d10` — one CSV
  line at 1 Hz. Both schemas are auto-detected by column count (14-col basic /
  23-col `lsi`), matching `BLE_CSV_HEADER` / `BLE_CSV_HEADER_LSI`.

No firmware change is needed — flash `nano_ble_telemetry` or `nano_lsi_log` as
usual, power the board on battery, and connect from the phone.

---

## What you need installed (one-time)

Node and Yarn are already assumed. The build also needs the **Android SDK + a
JDK 17** (this repo's machine only has JDK 25, which is too new for the Android
Gradle Plugin). Easiest path:

1. **Android Studio** — https://developer.android.com/studio
   Installs the Android SDK, `platform-tools` (`adb`), and a bundled **JDK 17**
   (its JBR). In *SDK Manager* make sure these are checked:
   - Android SDK Platform (latest, e.g. API 35/36)
   - Android SDK Platform-Tools
   - Android SDK Build-Tools
2. **Environment variables** (PowerShell, then restart the terminal):
   ```powershell
   setx ANDROID_HOME "$env:LOCALAPPDATA\Android\Sdk"
   setx JAVA_HOME "C:\Program Files\Android\Android Studio\jbr"
   # add platform-tools to PATH for adb:
   setx PATH "$env:PATH;$env:LOCALAPPDATA\Android\Sdk\platform-tools"
   ```
   (Or install Temurin **17** and point `JAVA_HOME` there.)
3. **Phone in developer mode** — enable *Developer options* → *USB debugging*,
   plug it in over USB, and accept the "Allow USB debugging" prompt. Verify:
   ```powershell
   adb devices        # your phone should be listed as "device"
   ```

> The Android **emulator cannot do BLE** — it has no Bluetooth radio. You must
> deploy to the physical phone on USB (which is why we build to a real device).

## Build & run

The **JavaScript side runs on any OS** — `yarn install` then `yarn start`
(`expo start`) serves Metro for development, and `yarn expo export` bundles it.

The **native Android APK** is where the OS matters. On **Linux** it builds cleanly;
on **Windows** the New-Architecture native build currently fails — `expo-modules-core`
and `react-native-reanimated`'s CMake step loops with
`ninja: error: manifest 'build.ninja' still dirty after 100 tries` (a Windows-only
prefab/ninja bug: the `react-native-workletsConfigVersion.cmake` phony output is not
present when ninja evaluates the manifest). So build the APK on Linux:

```bash
# on Linux / WSL (needs JDK 17, Android SDK + NDK 27.1.12297006 + cmake 3.22.1):
yarn install
yarn expo prebuild --platform android --clean          # generates android/ from app.json
cd android && ./gradlew assembleDebug -PreactNativeArchitectures=arm64-v8a
#   -> android/app/build/outputs/apk/debug/app-debug.apk   (~58 MB, arm64)
adb install -r app/build/outputs/apk/debug/app-debug.apk   # onto the USB phone
```

CI already builds the native target on its Linux runner. On **Windows**, develop
against Metro (`yarn start`) and `adb install` a Linux-built APK — the phone runs
the same bytecode either way. (`yarn android` / `expo run:android` works on Linux;
on Windows it hits the CMake bug above.)

> If `yarn install` ever complains about version mismatches, realign to the
> installed Expo SDK with `yarn expo install --fix`, then rebuild.

## Use it

1. Power the board (USB or battery) with `nano_ble_telemetry` / `nano_lsi_log`
   flashed — it advertises as `dtfit-gps`.
2. Open **dtfit monitor** on the phone, tap **CONNECT**, and grant the Bluetooth
   permission the first time.
3. It scans, connects, and streams. Read it at a glance:
   - **Banner** — green `GPS FIX`, amber `ACQUIRING FIX`, or red `NO DATA`
     (the board went silent / out of range).
   - **"Xs ago"** — the staleness clock; at 1 Hz it should read < 1 s. If it
     climbs past 3 s the link stalled.
   - **|a| ≈ 1.000 g** at rest confirms the IMU is alive even before GPS lock.
   - With `nano_lsi_log`: **MODE** (still/moving), per-update **COST (µs)**, and
     **EST ERR** (on-MCU float32 estimate vs raw fix, metres).

After the first `expo run:android`, day-to-day you can just relaunch the
installed app; rebuild only when the native config or dependencies change.

## Layout

```
mobile/dtfit-monitor/
├── App.tsx                 # the dashboard UI (Tailwind via NativeWind: className="...")
├── index.ts                # Expo entry (imports global.css)
├── app.json                # Expo config + Android BLE permissions (ble-plx plugin)
├── package.json            # Expo SDK 57 / RN 0.86 / ble-plx 3.5 / nativewind 4 / reanimated 4
├── babel.config.js         # babel-preset-expo with jsxImportSource: "nativewind"
├── metro.config.js         # Expo Metro config wrapped with withNativeWind
├── tailwind.config.js      # Tailwind v3 config: the dashboard palette (bg-/text-/border- colours)
├── global.css              # @tailwind base/components/utilities (compiled by Metro)
├── nativewind-env.d.ts     # NativeWind className types + a *.css module declaration
└── src/
    ├── telemetry.ts        # CSV parsing + schema detection (mirrors backend.py headers)
    └── useBleTelemetry.ts  # scan / connect / MTU / notify hook (UUIDs match the firmware)
```

Styling is **Tailwind** through [NativeWind](https://www.nativewind.dev) v4
(`className="..."`) — a Babel + Metro transform (`jsxImportSource: "nativewind"`
in `babel.config.js` plus `withNativeWind` in `metro.config.js`) that maps
Tailwind classes to React Native styles. The palette lives in `tailwind.config.js`,
so class names like `bg-card` / `text-ok` / `border-line` resolve the dashboard
colours (the same names the earlier twrnc setup used).

NativeWind's runtime import-references `react-native-reanimated`, so the app now
carries `react-native-reanimated` + `react-native-worklets` (native C++) alongside
`react-native-ble-plx` (Kotlin). The dashboard drives **no** animated utilities —
reanimated is a bundle-time dependency of NativeWind's runtime, not a feature this
screen uses — but Metro still needs it resolvable, so it is a real dependency.

Those native modules build fine on **Linux**; on **Windows** the New-Architecture
CMake step (`expo-modules-core` + `reanimated`) trips a prefab/ninja bug — see
[Build & run](#build--run). This is **not** the old `MAX_PATH` problem (that is
fixed: the repo is at a short path with `LongPathsEnabled=1`, and `worklets` + the
whole JS bundle build on Windows); it is a separate, Windows-only issue.

## Notes / troubleshooting

- **Nothing found when scanning?** Confirm the board prints
  `BLE advertising as 'dtfit-gps'` over USB serial, the phone's Bluetooth is on,
  and you granted the permission (Android 12+ needs *Nearby devices*).
- **Truncated / garbled line?** The app raises the MTU to 512 B on connect so the
  full `lsi` line fits one notification; if you see cut-off values, a stale build
  is running — reinstall with `yarn android`.
- **`ninja: error: manifest 'build.ninja' still dirty after 100 tries` on Windows?**
  This is the New-Architecture prefab bug (see [Build & run](#build--run)), not a
  path-length problem — raising `CMAKE_OBJECT_PATH_MAX` makes it worse. Build the
  APK on Linux/WSL and `adb install` it; use Windows only for Metro dev.
- **iPhone?** This is an Android build. iOS also works with Expo + ble-plx but
  needs a Mac (Xcode) to compile; not set up here.
