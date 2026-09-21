/*
 * nano_lsi_replay - a recorded rig drive replayed through the on-chip filter.
 *
 * The drive's new GPS fixes, as local east / north metres and seconds from the
 * first fix (replay_vec.h, from make_replay.py), go through two LsiFilter
 * instances exactly as nano_lsi_log feeds them, without the zero-velocity
 * update. Each update is printed as the raw float32 bits of both estimates so
 * the host compares them with a float64 run of the same algebra, and the
 * per-update cost of the two-axis step is accumulated from the DWT counter.
 *
 * Board:   Arduino Nano 33 BLE Sense Rev2 (arduino:mbed_nano:nano33ble)
 * Monitor: 115200 baud
 */
#include "dtfit_lsi.h"
#include "replay_vec.h"

static inline void dwtEnable() {
  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
  DWT->CYCCNT = 0;
  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}
static inline uint32_t dwtCycles() { return DWT->CYCCNT; }

LsiFilter fE, fN;

static void hex32(float f) {
  union { float f; uint32_t u; } v;
  v.f = f;
  char b[9];
  snprintf(b, sizeof b, "%08lX", (unsigned long)v.u);
  Serial.print(b);
}

void setup() {
  Serial.begin(115200);
  unsigned long t0 = millis();
  while (!Serial && millis() - t0 < 15000) { }
  dwtEnable();

  Serial.println();
  Serial.println("=== LSI on-MCU replay (float32) ===");
  Serial.print("config: W="); Serial.print(LSI_W);
  Serial.print(" order="); Serial.print(LSI_ORDER);
  Serial.print(" params="); Serial.print(LSI_N);
  Serial.print(" samples="); Serial.print(RP_N);
  Serial.print(" state="); Serial.print((unsigned)(2 * sizeof(LsiFilter)));
  Serial.println(" bytes");

  float pe[LSI_N] = {0}, pn[LSI_N] = {0};
  fE.reset(pe); fN.reset(pn);

  uint64_t cycSum = 0; uint32_t cycMax = 0; uint32_t nUpd = 0;
  for (int i = 0; i < RP_N; i++) {
    uint32_t c0 = dwtCycles();
    bool ue = fE.update(RP_T[i], RP_E[i]);
    bool un = fN.update(RP_T[i], RP_NN[i]);
    uint32_t dc = dwtCycles() - c0;
    if (ue && un) {
      cycSum += dc; if (dc > cycMax) cycMax = dc; nUpd++;
      Serial.print("R "); Serial.print(i); Serial.print(' ');
      hex32(fE.p[0]); Serial.print(' '); hex32(fE.p[1]); Serial.print(' ');
      hex32(fN.p[0]); Serial.print(' '); hex32(fN.p[1]); Serial.println();
    }
  }
  Serial.print("updates="); Serial.println(nUpd);
  if (nUpd) {
    float avg = (float)(cycSum / nUpd);
    Serial.print("cost: "); Serial.print(avg, 0);
    Serial.print(" cyc per two-axis update avg, ");
    Serial.print(avg / (LSI_F_CPU_HZ / 1e6f), 2); Serial.print(" us avg, ");
    Serial.print((float)cycMax / (LSI_F_CPU_HZ / 1e6f), 2); Serial.println(" us max");
  }
  Serial.println("=== end replay ===");
}

void loop() { delay(1000); }
