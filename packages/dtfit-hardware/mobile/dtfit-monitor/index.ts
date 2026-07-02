import { registerRootComponent } from "expo";

// NativeWind: register the compiled Tailwind styles before the app mounts.
import "./global.css";
import App from "./App";

// registerRootComponent calls AppRegistry.registerComponent('main', () => App)
// and wires up the Expo entry so the app boots in both dev and release builds.
registerRootComponent(App);
