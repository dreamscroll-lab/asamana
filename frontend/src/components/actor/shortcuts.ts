import { ACTION_IDENTITY } from "../../lib/actionIdentity";
import { ActionType, type ActionTypeValue } from "../../lib/contract";
import type { ActorOperation, ActorOperationCapability, ActorOperationParameters } from "../../types";

// Take, use and destroy are kinds of physical action, so they carry its icon: the card and the
// feed's record of the same deed then look alike.
const PHYSICAL_ICON = ACTION_IDENTITY[ActionType.physical].icon;

// Total over the operations on purpose, like ACTION_IDENTITY: one missing would show a bare label.
const LABELS: Record<ActorOperation | "idle", string> = {
  ...(Object.fromEntries(Object.entries(ACTION_IDENTITY).map(([type, { icon, label }]) => [type, `${icon} ${label}`])) as Record<ActionTypeValue, string>),
  broadcast: "📣 广播",
  take: `${PHYSICAL_ICON} 拿起`, use: `${PHYSICAL_ICON} 使用`, destroy: `${PHYSICAL_ICON} 毁坏`,
  idle: "⏸ 本步不行动",
};

export const OPERATION_LABEL: Record<string, string> = LABELS;

export function missingParameters(capability: ActorOperationCapability, parameters: ActorOperationParameters, description: string): boolean {
  return capability.required_parameters.some((key) => {
    const value = key === "description" ? description : parameters[key as keyof ActorOperationParameters];
    return value === undefined || value === "" || typeof value === "string" && !value.trim();
  });
}
