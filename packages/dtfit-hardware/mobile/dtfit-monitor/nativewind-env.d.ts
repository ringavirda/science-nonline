/// <reference types="nativewind/types" />

// The `import "./global.css"` entry is a Metro/NativeWind side-effect, not a JS
// module; declare it so `tsc` accepts the import.
declare module "*.css";
