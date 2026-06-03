/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        night: "#041425",
        ink: "#071d33",
        cyanGlow: "#33d4ff",
        trafficRed: "#ef4444",
        trafficYellow: "#facc15",
        trafficGreen: "#22c55e",
      },
      boxShadow: {
        panel: "0 30px 80px rgba(2, 12, 27, 0.55)",
        glow: "0 0 0 1px rgba(51, 212, 255, 0.12), 0 0 28px rgba(51, 212, 255, 0.12)",
      },
      backgroundImage: {
        dashboard:
          "radial-gradient(circle at top left, rgba(51,212,255,0.18), transparent 32%), radial-gradient(circle at top right, rgba(34,197,94,0.12), transparent 28%), linear-gradient(180deg, #05101f 0%, #081528 45%, #020817 100%)",
      },
    },
  },
  plugins: [],
};
