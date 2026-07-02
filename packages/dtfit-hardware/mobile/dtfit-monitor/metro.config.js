const { getDefaultConfig } = require("expo/metro-config");
const { withNativeWind } = require("nativewind/metro");

const config = getDefaultConfig(__dirname);

// Wire NativeWind's Tailwind CSS pipeline into Metro (compiles global.css).
module.exports = withNativeWind(config, { input: "./global.css" });
