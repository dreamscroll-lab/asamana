import { useState } from "react";
import { Outlet } from "react-router-dom";

import Sidebar from "./components/Sidebar";

const RAIL_KEY = "asamana.sidebar.collapsed";

/**
 * What the shell hands its routed view. Immersive is the view's state but the sidebar is the
 * shell's chrome, so the view reports it up. It overrides the stored choice without replacing
 * it: leaving immersive gives the user's own sidebar back.
 */
export type ShellContext = { setImmersive: (on: boolean) => void };

/**
 * The product shell: a collapsible world-list sidebar and the routed view beside it.
 *
 * Only the product's own screens sit inside it. The developer instruments are routed outside
 * (see main.tsx): each carries its own axis (a template, a world) in its header, so a sidebar
 * beside them would be a second world picker that picks nothing.
 */
export default function App() {
  // Stored, not just state: the dev instruments are routed outside the shell, so opening one
  // unmounts it. Read back on mount so the choice survives the trip (and a reload).
  const [collapsed, setCollapsed] = useState(() => {
    try {
      return localStorage.getItem(RAIL_KEY) === "1";
    } catch {
      return false;
    }
  });
  const toggle = () =>
    setCollapsed((c) => {
      try {
        localStorage.setItem(RAIL_KEY, c ? "0" : "1");
      } catch {
        /* no storage — the choice lives for this page only */
      }
      return !c;
    });

  const [immersive, setImmersive] = useState(false);

  return (
    <div className="h-screen flex overflow-hidden bg-slate-950 text-slate-100">
      <Sidebar collapsed={collapsed || immersive} onToggle={toggle} />
      <main className="flex-1 min-w-0 h-screen overflow-y-auto">
        <Outlet context={{ setImmersive } satisfies ShellContext} />
      </main>
    </div>
  );
}
