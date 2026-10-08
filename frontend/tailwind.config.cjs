/** @type {import('tailwindcss').Config} */

// Brand purple ramp. indigo/violet/purple utilities all map onto it, so the whole UI
// shares one accent — main #6B4EAA at 600, soft #8E72CC at 500.
const brand = {
  50: "#f4f0fb",
  100: "#eae3f7",
  200: "#d6c9ef",
  300: "#bda6e3",
  400: "#a689d8",
  500: "#8e72cc", // soft
  600: "#6b4eaa", // main
  700: "#573f8a",
  800: "#453268",
  900: "#35284f",
  950: "#221936",
};

// Neutral ramp biased toward the brand purple: a low-saturation violet-gray (hue ~262)
// instead of Tailwind's blue-black `slate`, which clashes with the accent. Overriding
// `slate` reskins every ground/panel/border/muted-text at once.
const neutral = {
  50: "#f8f5fc",
  100: "#eee9f5",
  200: "#ddd6ea",
  300: "#c4bcd6",
  400: "#a094ba", // muted text
  500: "#7d7099", // muted text
  600: "#5c4f7e",
  700: "#463a68", // subtle borders
  800: "#322850", // borders / hover fills
  900: "#201936", // panels
  950: "#150f24", // darkest ground
};

module.exports = {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  // Preflight off: styles.css is the bare-element base (box-sizing, form-control resets, the dark
  // ground); preflight would restyle every element a view leaves unstyled.
  corePlugins: { preflight: false },
  theme: {
    extend: {
      colors: { indigo: brand, violet: brand, purple: brand, slate: neutral },
    },
  },
  plugins: [],
};
