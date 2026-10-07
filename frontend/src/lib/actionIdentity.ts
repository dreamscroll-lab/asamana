/**
 * Each action type's icon and name, one table for every view so the map and the feed show a deed alike.
 *
 * Typed as total over the action types on purpose: keyed by `string`, a newly added type would fall
 * through to a placeholder unnoticed.
 */

import { ActionType, type ActionTypeValue } from "./contract";

export interface ActionIdentity {
  icon: string;
  label: string;
}

export const ACTION_IDENTITY: Record<ActionTypeValue, ActionIdentity> = {
  [ActionType.talk]:         { icon: "💬", label: "交谈" },
  [ActionType.move]:         { icon: "🚶", label: "移动" },
  [ActionType.physical]:     { icon: "💪", label: "动手" },
  [ActionType.covert]:       { icon: "🥷", label: "潜行" },
  [ActionType.work]:         { icon: "🔨", label: "行事" },
  [ActionType.rest]:         { icon: "💤", label: "歇息" },
  [ActionType.send_message]: { icon: "📨", label: "传信" },
  [ActionType.errand]:       { icon: "🏃", label: "派人" },
};
