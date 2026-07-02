/**
 * MapScreen -- live map of the rig's reported position (the app's second tab).
 *
 * A Leaflet map (keyless CARTO tiles -- no Google Maps API key) inside a WebView.
 * React Native injects each new lat/lon so the marker follows the rig and a
 * breadcrumb trail draws the track; you can eyeball whether the fix lands on the
 * real road. The `dark` prop swaps CARTO dark<->light tiles (dark washes out in
 * bright sun). Tiles need internet (works on cellular during a drive); offline,
 * the map is blank but the coord readout still updates.
 */
import { useEffect, useRef } from "react";
import { Text, View } from "react-native";
import { WebView } from "react-native-webview";

// Self-contained map page. Leaflet + tiles load from CDN (same internet the tiles
// need); setPos()/setTheme() are called from RN via injectJavaScript.
const MAP_HTML = `<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  html,body,#map{height:100%;margin:0;background:#0b0f14;}
  .leaflet-container{background:#0b0f14;}
  .leaflet-control-attribution{font-size:9px;background:rgba(127,127,127,0.25);color:#666;}
  .leaflet-control-attribution a{color:#2f81f7;}
</style>
</head>
<body>
<div id="map"></div>
<script>
  var DARK='https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png';
  var LIGHT='https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png';
  var map = L.map('map', {zoomControl:true, attributionControl:true}).setView([50.45,30.45], 15);
  var tiles = L.tileLayer(DARK, {
    maxZoom:19, subdomains:'abcd',
    attribution:'&copy; OpenStreetMap &copy; CARTO'
  }).addTo(map);
  var trail = L.polyline([], {color:'#2f81f7', weight:3, opacity:0.9}).addTo(map);
  var marker = null, first = true, stroke = '#e6edf3';
  window.setTheme = function(dark){
    tiles.setUrl(dark ? DARK : LIGHT);
    var bg = dark ? '#0b0f14' : '#eef1f5';
    document.body.style.background = bg;
    document.getElementById('map').style.background = bg;
    stroke = dark ? '#e6edf3' : '#1f2328';
    if (marker) marker.setStyle({color: stroke});
  };
  window.setPos = function(lat, lon, fix){
    if (!isFinite(lat) || !isFinite(lon) || (lat===0 && lon===0)) return;
    var ll = [lat, lon];
    var col = fix ? '#2ea043' : '#d29922';
    if (!marker) {
      marker = L.circleMarker(ll, {radius:8, color:stroke, weight:2, fillColor:col, fillOpacity:1}).addTo(map);
    } else {
      marker.setLatLng(ll).setStyle({fillColor:col});
    }
    trail.addLatLng(ll);
    if (first) { map.setView(ll, 17); first=false; } else { map.panTo(ll, {animate:true}); }
  };
  if (window.ReactNativeWebView) window.ReactNativeWebView.postMessage('ready');
</script>
</body>
</html>`;

export default function MapScreen({
  lat,
  lon,
  fix,
  live,
  dark,
}: {
  lat?: number;
  lon?: number;
  fix?: number;
  live: boolean;
  dark: boolean;
}) {
  const ref = useRef<WebView>(null);

  const has =
    lat != null &&
    lon != null &&
    Number.isFinite(lat) &&
    Number.isFinite(lon) &&
    !(lat === 0 && lon === 0);

  const push = () => {
    if (!has) return;
    ref.current?.injectJavaScript(
      `window.setPos && window.setPos(${lat}, ${lon}, ${fix ?? 0}); true;`,
    );
  };

  const applyTheme = () => {
    ref.current?.injectJavaScript(`window.setTheme && window.setTheme(${dark}); true;`);
  };

  // Feed each new position into the map (skipped silently until Leaflet loads;
  // onMessage('ready') re-applies theme + re-pushes the latest so it appears at once).
  useEffect(push, [lat, lon, fix]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(applyTheme, [dark]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <View className="flex-1 bg-bg">
      <WebView
        ref={ref}
        source={{ html: MAP_HTML }}
        originWhitelist={["*"]}
        javaScriptEnabled
        domStorageEnabled
        onMessage={() => {
          applyTheme();
          push();
        }}
        // Some Android WebViews start throttled in the background tab; keep it live.
        androidLayerType="hardware"
        style={{ flex: 1, backgroundColor: dark ? "#0b0f14" : "#eef1f5" }}
      />
      <View className="absolute top-3 left-3 right-3 flex-row justify-between items-start">
        <View
          className={`flex-row items-center border rounded-full px-3 py-1.5 bg-card ${
            live ? "border-ok" : "border-line"
          }`}
        >
          <View
            className={`w-2 h-2 rounded-full mr-1.5 ${live ? "bg-ok" : "bg-dim"}`}
          />
          <Text className="text-ink text-xs font-semibold">
            {has
              ? `${lat!.toFixed(6)}, ${lon!.toFixed(6)}`
              : "waiting for fix..."}
          </Text>
        </View>
      </View>
    </View>
  );
}
