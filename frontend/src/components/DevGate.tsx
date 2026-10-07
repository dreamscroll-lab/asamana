import { Link } from "react-router-dom";

import { useDevTools } from "../lib/deployment";

/**
 * A developer screen, shown only when the backend mounted the routes it reads. Off, the
 * screen is not loaded and `off` says why, instead of every panel failing on its own 404.
 */
export default function DevGate({ off, children }: { off: string; children: React.ReactNode }) {
  const on = useDevTools();
  if (on === null) return <div className="h-screen bg-slate-950" />;
  if (!on) {
    return (
      <div className="h-screen grid place-items-center bg-slate-950 px-8 text-center">
        <div className="space-y-3">
          <p className="text-sm text-slate-400">{off}</p>
          <Link to="/" className="text-xs text-slate-500 hover:text-slate-200 transition">
            ‹ Asamana
          </Link>
        </div>
      </div>
    );
  }
  return <>{children}</>;
}
