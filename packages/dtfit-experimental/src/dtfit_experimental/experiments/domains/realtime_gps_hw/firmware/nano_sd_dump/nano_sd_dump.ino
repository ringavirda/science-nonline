/*
 * nano_sd_dump - stream a session file off the SD card over USB serial. Since nano_lsi_log v3
 * writes a fresh per-session file (YYYYMMDD_HHMMSS.CSV, or SESSnnnn.CSV before a GPS fix), this
 * first prints a directory LISTING, then auto-dumps the NEWEST .csv (by FAT modify-time, which
 * the logger stamps from GPS UTC). The host extracts the bytes between the markers.
 * Prints:  <<<LIST>>>  name size fatdate fattime  (one per line)  <<<ENDLIST>>>
 *          <<<BEGIN name size>>>  <raw file bytes>  <<<END>>>
 *
 * Board:   Arduino Nano 33 BLE Sense Rev2.  Library: SdFat.  Monitor: 115200.
 */
#include <SPI.h>
#include "SdFat.h"
#include <string.h>

#define SD_CS 10
#define MAX_YEAR 2027         // files dated later are spoof-clock junk; skip when auto-picking
SdFs sd;
FsFile root, f;

static bool isCsv(const char* n) {
  int L = strlen(n);
  return L > 4 && n[L - 4] == '.' &&
         (n[L - 3] == 'c' || n[L - 3] == 'C') &&
         (n[L - 2] == 's' || n[L - 2] == 'S') &&
         (n[L - 1] == 'v' || n[L - 1] == 'V');
}

void setup() {
  Serial.begin(115200);
  unsigned long t0 = millis();
  while (!Serial && millis() - t0 < 10000) { }

  if (!sd.begin(SdSpiConfig(SD_CS, SHARED_SPI, SD_SCK_MHZ(4)))) {
    Serial.println("<<<ERR no SD>>>");
    return;
  }

  char name[64], newest[64] = {0};
  uint32_t newestKey = 0;                 // (fatDate << 16) | fatTime -> newest wins
  if (!root.open("/")) { Serial.println("<<<ERR no root>>>"); return; }

  Serial.println("<<<LIST>>>");
  while (f.openNext(&root, O_READ)) {
    if (!f.isDir()) {
      f.getName(name, sizeof(name));
      uint16_t d = 0, tm = 0;
      f.getModifyDateTime(&d, &tm);
      uint32_t sz = f.fileSize();
      Serial.print(name); Serial.print(' '); Serial.print(sz);
      Serial.print(' '); Serial.print(d); Serial.print(' '); Serial.println(tm);
      // pick the newest CSV, but IGNORE spoof-dated files: a GPS spoof falsifies the clock
      // (observed: a fake 2028 date), which would otherwise out-rank a real drive. FAT year =
      // 1980 + (date >> 9); a genuine run is within [2024, MAX_YEAR].
      int yr = 1980 + (d >> 9);
      uint32_t key = ((uint32_t)d << 16) | tm;
      if (isCsv(name) && yr >= 2024 && yr <= MAX_YEAR && key >= newestKey) {
        newestKey = key;
        strncpy(newest, name, sizeof(newest) - 1);
      }
    }
    f.close();
  }
  root.close();
  Serial.println("<<<ENDLIST>>>");

  if (!newest[0]) { Serial.println("<<<ERR no csv>>>"); return; }
  f = sd.open(newest, O_READ);
  if (!f) { Serial.println("<<<ERR open fail>>>"); return; }
  Serial.print("<<<BEGIN "); Serial.print(newest);
  Serial.print(' '); Serial.print(f.size()); Serial.println(">>>");
  uint8_t buf[64];
  while (f.available()) {
    int k = f.read(buf, sizeof(buf));
    if (k > 0) Serial.write(buf, k);
  }
  f.close();
  Serial.println();
  Serial.println("<<<END>>>");
}

void loop() { }
