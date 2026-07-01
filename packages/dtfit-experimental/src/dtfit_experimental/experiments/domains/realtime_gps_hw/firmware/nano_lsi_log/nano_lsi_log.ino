/*
 * nano_lsi_log - the rig's working firmware: on-MCU dtfit LSI on the live GPS,
 * logging BOTH the raw GPS+IMU stream AND the on-board filter estimate, to BLE
 * *and* the SD card at once (no field reflash). The PC scores the float32 on-MCU
 * result against its float64 reference.
 *
 * v2 -- two upgrades over the scalar v1 (c0+c1*t on raw lat/lon degrees):
 *
 *  1. LOCAL-ENU COORDS. The fix is converted to local east/north metres about the
 *     first-fix origin before filtering. Small metre values are well-conditioned in
 *     float32 (raw degrees ~50/30 hit the float32 floor -- the ~few-m precision drop
 *     we measured). The estimate is converted back to lat/lon only for the log.
 *
 *  2. IMU-ADAPTIVE MODEL. The model order is switched from the inertial readings,
 *     the dtfit "the filter carries the model, the model adapts to the regime" idea:
 *       - STILL  (gyro & accel at rest, GPS speed ~0): zero-velocity update (ZUPT) ->
 *         the degree-1 model collapses to degree-0 (c0), a position average; the
 *         forecast goes flat (no phantom drift-as-motion extrapolation).
 *       - MOVING: full degree-1 (c0 + c1*t), tracking velocity.
 *     Same filter, same tables -- the switch is a constraint applied to the public
 *     state, not a second method. ``mode`` (0 still / 1 moving) is logged.
 *
 * v3 -- one SD file PER SESSION: each boot opens a fresh SESSnnnn.CSV and renames it to
 *   YYYYMMDD_HHMMSS.CSV once GPS UTC is known (no more appending to a single riglog.csv), so
 *   a power loss can never touch a prior session. Each line is still sync()'d immediately, and
 *   FAT modify-times are stamped from GPS so the host can pull the newest session.
 *
 * Record (23 cols):
 *   t_ms,sats,fix,lat,lon,alt_m,hdop,spd_kmph,ax,ay,az,gx,gy,gz,mx,my,mz,
 *   est_lat,est_lon,fc_lat,fc_lon,mode,cost_us
 *
 * Board:   Arduino Nano 33 BLE Sense Rev2 (arduino:mbed_nano:nano33ble)
 * Library: ArduinoBLE, TinyGPSPlus, Arduino_BMI270_BMM150, SdFat
 * Monitor: 115200 baud
 */
#include <ArduinoBLE.h>
#include <TinyGPS++.h>
#include "Arduino_BMI270_BMM150.h"
#include <SPI.h>
#include "SdFat.h"
#include "dtfit_lsi.h"
#include <math.h>

#define SVC_UUID  "9a1e0000-1b2c-4f3a-8d5e-6f7a8b9c0d10"
#define TELE_UUID "9a1e0001-1b2c-4f3a-8d5e-6f7a8b9c0d10"
BLEService telemetry(SVC_UUID);
BLEStringCharacteristic teleChar(TELE_UUID, BLERead | BLENotify, 200);

#define SD_CS 10
SdFs sd;
FsFile sdLog;
bool sdOk = false;

static const char* HEADER =
  "t_ms,sats,fix,lat,lon,alt_m,hdop,spd_kmph,ax,ay,az,gx,gy,gz,mx,my,mz,"
  "est_lat,est_lon,fc_lat,fc_lon,mode,cost_us";

static const double M_PER_DEG = 111320.0;   // metres per degree of latitude

TinyGPSPlus gps;

// --- per-session logging: a fresh timestamped file each boot (no append / no rewrite) ---
char logName[24] = {0};      // current SD file name for this session
bool logNamed = false;       // renamed from SESSnnnn to the GPS-UTC name yet?

// FAT modify-time from GPS UTC, so each file carries a real timestamp and the host can pick
// the newest session; falls back to a fixed date before the first fix.
static void gpsFatDateTime(uint16_t* date, uint16_t* time) {
  if (gps.date.isValid() && gps.time.isValid() && gps.date.year() >= 2021) {
    *date = FS_DATE(gps.date.year(), gps.date.month(), gps.date.day());
    *time = FS_TIME(gps.time.hour(), gps.time.minute(), gps.time.second());
  } else {
    *date = FS_DATE(2026, 1, 1);
    *time = FS_TIME(0, 0, 0);
  }
}

// Once GPS UTC date/time is valid, rename this session's file from its boot-time SESSnnnn
// placeholder to YYYYMMDD_HHMMSS.CSV -- done once, early (before real motion), so the run then
// appends to a stable open file. No fix ever -> stays SESSnnnn (still unique, no data lost).
static void maybeNameLog() {
  if (!sdOk || logNamed || !sdLog) return;
  if (!(gps.date.isValid() && gps.time.isValid() && gps.date.year() >= 2021)) return;
  char ts[24];
  snprintf(ts, sizeof(ts), "%04u%02u%02u_%02u%02u%02u.CSV",
           gps.date.year(), gps.date.month(), gps.date.day(),
           gps.time.hour(), gps.time.minute(), gps.time.second());
  logNamed = true;                       // attempt once regardless of outcome
  if (sd.exists(ts)) return;             // same-second file exists -> keep SESSnnnn, don't clobber
  if (sdLog.rename(ts)) {
    strncpy(logName, ts, sizeof(logName) - 1);
    sdLog.sync();
    Serial.print("SD: session file -> "); Serial.println(logName);
  }
}

LsiFilter fE, fN;            // degree-1 LSI on local East / North metres
bool   originSet = false;
double lat0 = 0, lon0 = 0;   // ENU origin (first fix)
double cosLat0 = 1.0;
unsigned long gStart = 0;
int  staticCount = 0;        // debounce for the rest detector
bool prevStatic = false;

static inline void dwtEnable() {
  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
  DWT->CYCCNT = 0;
  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}
static inline uint32_t dwtCycles() { return DWT->CYCCNT; }

// Zero-velocity update: pin the velocity parameter to 0 and decouple its
// covariance, so the degree-1 model behaves as degree-0 (a constant) while still.
static void zupt(LsiFilter& f) {
  f.p[1] = 0.0f;
  f.P[0][1] = 0.0f; f.P[1][0] = 0.0f; f.P[1][1] = 1e-6f;
}
// Release the velocity (re-inflate its covariance) when motion resumes, so the
// filter can re-acquire a velocity quickly instead of staying pinned near zero.
static void release(LsiFilter& f) {
  f.P[1][1] = LSI_P0_DIAG;
  f.P[0][1] = 0.0f; f.P[1][0] = 0.0f;
}

static String buildLine() {
  float ax = 0, ay = 0, az = 0, gx = 0, gy = 0, gz = 0, mx = 0, my = 0, mz = 0;
  if (IMU.accelerationAvailable()) IMU.readAcceleration(ax, ay, az);
  if (IMU.gyroscopeAvailable())    IMU.readGyroscope(gx, gy, gz);
  if (IMU.magneticFieldAvailable()) IMU.readMagneticField(mx, my, mz);  // BMM150 (uT)
  int sats = gps.satellites.isValid() ? gps.satellites.value() : 0;
  int fix = gps.location.isValid() ? 1 : 0;

  // --- rest detector from the inertial readings (+ GPS speed corroboration) ---
  float gMag = sqrtf(gx * gx + gy * gy + gz * gz);          // deg/s
  float aMag = sqrtf(ax * ax + ay * ay + az * az);          // g (~1 at rest)
  float spd = fix ? gps.speed.kmph() : 0.0f;
  bool restNow = (gMag < 3.0f) && (fabsf(aMag - 1.0f) < 0.05f) && (spd < 2.0f);
  if (restNow) { if (staticCount < 5) staticCount++; } else staticCount = 0;
  bool isStatic = (staticCount >= 3);

  double estLat = 0, estLon = 0, fcLat = 0, fcLon = 0;
  float costUs = 0; int mode = isStatic ? 0 : 1;
  if (fix) {
    double lat = gps.location.lat(), lon = gps.location.lng();
    if (!originSet) {
      lat0 = lat; lon0 = lon; cosLat0 = cos(lat0 * M_PI / 180.0);
      float pe[LSI_N] = {0}, pn[LSI_N] = {0};   // origin -> ENU (0,0)
      fE.reset(pe); fN.reset(pn);
      gStart = millis(); originSet = true;
    }
    // local ENU metres about the origin (small -> float32-safe)
    float e = (float)((lon - lon0) * cosLat0 * M_PER_DEG);
    float n = (float)((lat - lat0) * M_PER_DEG);
    float t = (millis() - gStart) / 1000.0f;

    if (prevStatic && !isStatic) { release(fE); release(fN); }   // motion resumed

    uint32_t c0 = dwtCycles();
    fE.update(t, e);
    fN.update(t, n);
    if (isStatic) { zupt(fE); zupt(fN); }      // still -> degree-0 (ZUPT)
    uint32_t dc = dwtCycles() - c0;
    costUs = (float)dc / (LSI_F_CPU_HZ / 1e6f);

    float estE = fE.predict(t), estN = fN.predict(t);
    float fcE = isStatic ? estE : fE.predict(t + 1.0f);
    float fcN = isStatic ? estN : fN.predict(t + 1.0f);
    // back to lat/lon for the log (large origin added in double precision)
    estLat = lat0 + (double)estN / M_PER_DEG;
    estLon = lon0 + (double)estE / (M_PER_DEG * cosLat0);
    fcLat  = lat0 + (double)fcN / M_PER_DEG;
    fcLon  = lon0 + (double)fcE / (M_PER_DEG * cosLat0);
    prevStatic = isStatic;
  }

  String s = String(millis());
  s += ","; s += sats;
  s += ","; s += fix;
  s += ","; s += String(gps.location.lat(), 6);
  s += ","; s += String(gps.location.lng(), 6);
  s += ","; s += String(gps.altitude.meters(), 1);
  s += ","; s += String(gps.hdop.hdop(), 1);
  s += ","; s += String(gps.speed.kmph(), 1);
  s += ","; s += String(ax, 3); s += ","; s += String(ay, 3); s += ","; s += String(az, 3);
  s += ","; s += String(gx, 2); s += ","; s += String(gy, 2); s += ","; s += String(gz, 2);
  s += ","; s += String(mx, 1); s += ","; s += String(my, 1); s += ","; s += String(mz, 1);
  s += ","; s += String(estLat, 6); s += ","; s += String(estLon, 6);
  s += ","; s += String(fcLat, 6);  s += ","; s += String(fcLon, 6);
  s += ","; s += mode;
  s += ","; s += String(costUs, 1);
  return s;
}

void setup() {
  Serial.begin(115200);
  unsigned long t0 = millis();
  while (!Serial && millis() - t0 < 8000) { }
  Serial1.begin(9600);
  IMU.begin();
  dwtEnable();

  sdOk = sd.begin(SdSpiConfig(SD_CS, SHARED_SPI, SD_SCK_MHZ(4)));
  if (sdOk) {
    FsDateTime::setCallback(gpsFatDateTime);
    int idx = 0;                          // a fresh, unique file per boot
    do { snprintf(logName, sizeof(logName), "SESS%04d.CSV", idx++); }
    while (sd.exists(logName) && idx < 10000);
    sdLog = sd.open(logName, O_WRITE | O_CREAT | O_TRUNC);
    if (sdLog) { sdLog.println(HEADER); sdLog.sync(); }
    else sdOk = false;
  }
  Serial.print(sdOk ? "SD: logging to " : "SD: not present (BLE/serial only)");
  Serial.println(sdOk ? logName : "");

  if (!BLE.begin()) {
    Serial.println("BLE init FAILED");
  } else {
    BLE.setDeviceName("dtfit-gps");
    BLE.setLocalName("dtfit-gps");
    BLE.setAdvertisedService(telemetry);
    telemetry.addCharacteristic(teleChar);
    BLE.addService(telemetry);
    teleChar.writeValue("boot");
    BLE.advertise();
    Serial.println("BLE advertising as 'dtfit-gps'");
  }
  Serial.println("model: ENU-LSI, IMU-adaptive (mode 0=still/ZUPT degree-0, 1=moving degree-1)");
  Serial.println(HEADER);
}

static void emit(const String& line) {
  Serial.println(line);
  if (sdOk && sdLog) { sdLog.println(line); sdLog.sync(); }
  teleChar.writeValue(line);
}

void loop() {
  static unsigned long last = 0;
  BLE.central();
  while (Serial1.available()) gps.encode(Serial1.read());
  if (millis() - last >= 1000) {
    last = millis();
    emit(buildLine());
    maybeNameLog();          // rename SESSnnnn -> GPS-UTC name once, at first fix
  }
}
