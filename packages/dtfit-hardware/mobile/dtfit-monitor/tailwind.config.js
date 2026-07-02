/** @type {import('tailwindcss').Config} */
// NativeWind (Tailwind for React Native) config. The rig dashboard palette lives
// here as `theme.extend.colors`, so classes like `bg-card` / `text-ok` /
// `border-line` resolve the same names the UI already used under twrnc.
module.exports = {
  content: ["./App.tsx", "./index.ts", "./src/**/*.{ts,tsx}"],
  presets: [require("nativewind/preset")],
  theme: {
    extend: {
      // Colours resolve through CSS variables (defined in global.css) so the whole UI
      // re-themes light/dark without touching a single className -- only the variable
      // values swap. See global.css for the light (default) + dark (@media) palettes.
      colors: {
        bg: "rgb(var(--c-bg) / <alpha-value>)",
        card: "rgb(var(--c-card) / <alpha-value>)",
        card2: "rgb(var(--c-card2) / <alpha-value>)",
        line: "rgb(var(--c-line) / <alpha-value>)",
        ink: "rgb(var(--c-ink) / <alpha-value>)",
        dim: "rgb(var(--c-dim) / <alpha-value>)",
        ok: "rgb(var(--c-ok) / <alpha-value>)",
        warn: "rgb(var(--c-warn) / <alpha-value>)",
        bad: "rgb(var(--c-bad) / <alpha-value>)",
        info: "rgb(var(--c-info) / <alpha-value>)",
        accent: "rgb(var(--c-accent) / <alpha-value>)",
      },
    },
  },
  plugins: [],
};
