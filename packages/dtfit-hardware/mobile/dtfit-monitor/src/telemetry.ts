/**
 * Telemetry parsing for the realtime_gps_hw rig.
 *
 * The board notifies one CSV line at 1 Hz on characteristic 9a1e0001-...  Two
 * firmwares share the same GATT service and differ only in column count, so we
 * detect the schema by field count:
 *
 *   basic (14 cols, nano_ble_telemetry):
 *     t_ms,sats,fix,lat,lon,alt_m,hdop,spd_kmph,ax,ay,az,gx,gy,gz
 *   lsi   (23/24 cols, nano_lsi_log -- adds mag + the on-MCU LSI estimate/forecast):
 *     ...,mx,my,mz,est_lat,est_lon,fc_lat,fc_lon,mode,cost_us[,sd]
 *     The optional 24th ``sd`` col (0/1) is the board's live SD-logging status;
 *     firmware without it parses as 23 cols and leaves ``sd`` undefined.
 *   lsi2  (27/28 cols, nano_lsi_log v4 -- adds the high-rate on-MCU heading block):
 *     ...,cost_us,hdg_deg,dhdg_deg,imu_hz,newfix[,sd]
 *     hdg_deg on-MCU heading (0-360): the v7 firmware makes this the COMPLEMENTARY-cleaned,
 *     GPS-course-anchored (drift-free) heading; pre-v7 it was the raw gyro integrator. Same
 *     column either way. dhdg_deg the RAW gyro yaw increment since the last emit; imu_hz IMU
 *     samples integrated this interval; newfix 1 on a fresh GPS fix else 0.
 *     Reported as schema "lsi" too (a superset), with the extra fields populated.
 *
 * These headers mirror backend.py's BLE_CSV_HEADER / BLE_CSV_HEADER_LSI.
 */

export type Schema = "basic" | "lsi";

export interface Telemetry {
  schema: Schema;
  raw: string;
  tMs: number;
  sats: number;
  fix: number;
  lat: number;
  lon: number;
  altM: number;
  hdop: number;
  spdKmph: number;
  ax: number;
  ay: number;
  az: number;
  gx: number;
  gy: number;
  gz: number;
  // lsi schema only
  mx?: number;
  my?: number;
  mz?: number;
  estLat?: number;
  estLon?: number;
  fcLat?: number;
  fcLon?: number;
  mode?: number;
  costUs?: number;
  // lsi2 (v4) only: high-rate on-MCU heading block
  hdgDeg?: number; // on-MCU heading 0-360 (v7: complementary-cleaned/drift-free; pre-v7: raw gyro)
  dhdgDeg?: number; // RAW gyro yaw increment since the last emit
  imuHz?: number; // IMU samples integrated this interval
  newfix?: number; // 1 if this line carries a fresh GPS fix, 0 for a between-fix sample
  sd?: number; // recording status: 0 paused (OFF), 1 recording (REC), 2 recording but no write (NO SD)
}

const BASIC_COLS = 14;
const LSI_COLS = 23;
const LSI_SD_COLS = 24; // lsi record with the trailing `sd` status column
const LSI2_COLS = 27; // v4: lsi + hdg_deg,dhdg_deg,imu_hz,newfix
const LSI2_SD_COLS = 28; // v4 with the trailing `sd` status column

/**
 * Parse one telemetry line. Returns null for anything that is not a data row:
 * the "boot" sentinel the firmware writes on (re)connect, the header line, blank
 * lines, or a record with an unexpected column count / non-numeric timestamp.
 */
export function parseTelemetry(line: string): Telemetry | null {
  const s = line.trim();
  if (!s || s === "boot") return null;

  const parts = s.split(",");
  if (parts[0] === "t_ms") return null; // header echo
  const isLsi = parts.length === LSI_COLS || parts.length === LSI_SD_COLS;
  const isLsi2 = parts.length === LSI2_COLS || parts.length === LSI2_SD_COLS;
  if (parts.length !== BASIC_COLS && !isLsi && !isLsi2) return null;

  const n = parts.map((p) => Number(p));
  if (!Number.isFinite(n[0])) return null;

  const schema: Schema = isLsi || isLsi2 ? "lsi" : "basic";
  const t: Telemetry = {
    schema,
    raw: s,
    tMs: n[0],
    sats: n[1],
    fix: n[2],
    lat: n[3],
    lon: n[4],
    altM: n[5],
    hdop: n[6],
    spdKmph: n[7],
    ax: n[8],
    ay: n[9],
    az: n[10],
    gx: n[11],
    gy: n[12],
    gz: n[13],
  };
  if (schema === "lsi") {
    t.mx = n[14];
    t.my = n[15];
    t.mz = n[16];
    t.estLat = n[17];
    t.estLon = n[18];
    t.fcLat = n[19];
    t.fcLon = n[20];
    t.mode = n[21];
    t.costUs = n[22];
    if (isLsi2) {
      t.hdgDeg = n[23];
      t.dhdgDeg = n[24];
      t.imuHz = n[25];
      t.newfix = n[26];
      if (parts.length === LSI2_SD_COLS) t.sd = n[27]; // 0/1 live SD-logging status
    } else if (parts.length === LSI_SD_COLS) {
      t.sd = n[23]; // 0/1 live SD-logging status (old lsi record)
    }
  }
  return t;
}

/** Great-circle distance in metres between two lat/lon points (haversine). */
export function metresBetween(
  lat1: number,
  lon1: number,
  lat2: number,
  lon2: number,
): number {
  const R = 6371000;
  const toRad = (d: number) => (d * Math.PI) / 180;
  const dLat = toRad(lat2 - lat1);
  const dLon = toRad(lon2 - lon1);
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon / 2) ** 2;
  return 2 * R * Math.asin(Math.min(1, Math.sqrt(a)));
}

/** Accelerometer magnitude in g (should sit near 1.0 g at rest). */
export function accelMagnitude(t: Telemetry): number {
  return Math.sqrt(t.ax * t.ax + t.ay * t.ay + t.az * t.az);
}
