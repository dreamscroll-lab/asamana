import { ACTION_IDENTITY } from "../../lib/actionIdentity";
import type { ActorOperationCapability, ActorOperationParameters } from "../../types";

export const OPERATION_LABEL: Record<string, string> = {
  ...Object.fromEntries(Object.entries(ACTION_IDENTITY).map(([type, { icon, label }]) => [type, `${icon} ${label}`])),
  broadcast: "📣 广播", take: "拿起", use: "使用", destroy: "毁坏",
  idle: "⏸ 本步不行动",
};

export function missingParameters(capability: ActorOperationCapability, parameters: ActorOperationParameters, description: string): boolean {
  return capability.required_parameters.some((key) => {
    const value = key === "description" ? description : parameters[key as keyof ActorOperationParameters];
    return value === undefined || value === "" || typeof value === "string" && !value.trim();
  });
}
