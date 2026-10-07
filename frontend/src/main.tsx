import { lazy, StrictMode, Suspense } from "react";
import { createRoot } from "react-dom/client";
import { createHashRouter, RouterProvider } from "react-router-dom";

import App from "./App";
import WorldsProvider from "./WorldsProvider";
import BuildingScreen from "./components/BuildingScreen";
import DevGate from "./components/DevGate";
import HomeView from "./views/HomeView";
import ReviewView from "./views/ReviewView";
import WorldView from "./views/WorldView";
import "./tailwind.css";
import "./styles.css";

// Screens an ordinary visit never passes through are deferred. Must be: the map workbench mounts
// the real renderer, so a static import drags Phaser (~1.4 MB) and every fixture into the entry
// chunk and defeats WorldView's lazy `LiveWorldMap`.
//
// BuildingScreen is not deferred: HomeView imports it directly, so it is in the entry graph anyway.
const LabView = lazy(() => import("./views/LabView"));
const DevView = lazy(() => import("./views/DevView"));
const SplashScreen = lazy(() => import("./components/SplashScreen"));

// A deferred screen is the whole window: while its chunk lands, show bare ground, not substitute chrome.
const deferred = (node: React.ReactNode) => (
  <Suspense fallback={<div className="h-screen bg-slate-950" />}>{node}</Suspense>
);

// Hash routing keeps deep links working without an SPA-fallback route on the static host.
//
// `WorldsProvider` holds the world list read by both the shell and the developer tools. `App` is the
// shell (sidebar + view) for the product screens only; the instruments are its siblings because each
// picks its own template or world in its header, and a world sidebar there would only cost the map 256px.
const router = createHashRouter([
  {
    path: "/",
    element: <WorldsProvider />,
    children: [
      {
        element: <App />,
        children: [
          { index: true, element: <HomeView /> },
          // The build wait on its own, reachable by URL only: it plays only during a build, so this
          // is how to work on it without building a world each time.
          { path: "lab/loading", element: <BuildingScreen /> },
          { path: "worlds/:id/review", element: <ReviewView /> },
          { path: "worlds/:id", element: <WorldView /> },
        ],
      },
      // Map workbench: a template's map, its cast frame by frame, and the renderer
      // driven off fixtures, all read off the template's live files. Needs the backend's dev routes.
      {
        path: "lab",
        element: (
          <DevGate off="The map workbench is a developer tool and the backend has it off. Set ASAMANA_DEV_TOOLS=true and restart the backend.">
            {deferred(<LabView />)}
          </DevGate>
        ),
      },
      // "开发者工具" (developer tools): traces / audit / workbench. Deferred like the workbench.
      {
        path: "dev",
        element: (
          <DevGate off="Developer tools are off on this backend. Set ASAMANA_DEV_TOOLS=true and restart it.">
            {deferred(<DevView />)}
          </DevGate>
        ),
      },
    ],
  },
  // The launch screen is the whole window, so it sits outside the app shell. Reachable by URL only,
  // like lab/loading.
  { path: "/lab/splash", element: deferred(<SplashScreen />) },
]);

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <RouterProvider router={router} />
  </StrictMode>,
);
