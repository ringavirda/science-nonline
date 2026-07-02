module.exports = function (api) {
  api.cache(true);
  return {
    presets: [
      // NativeWind v4 drives className via its jsx-runtime -- selected with
      // jsxImportSource: "nativewind". No "nativewind/babel" preset is needed
      // (that legacy preset also re-adds the worklets plugin, duplicating it).
      // babel-preset-expo auto-injects react-native-worklets/plugin because
      // react-native-reanimated is installed -- NativeWind's runtime imports it.
      ["babel-preset-expo", { jsxImportSource: "nativewind" }],
    ],
  };
};
