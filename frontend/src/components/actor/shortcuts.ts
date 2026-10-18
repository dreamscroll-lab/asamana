import type { ActorOperationCapability, ActorOperationParameters } from "../../types";

export const OPERATION_LABEL: Record<string, string> = {
  move: "🚶 移动", broadcast: "📣 广播", talk: "💬 交谈",
  work: "🔨 行事", rest: "💤 歇息", covert: "🥷 潜行",
  errand: "派人", take: "拿起", use: "使用", destroy: "毁坏", physical: "动手",
};

export function missingParameters(capability: ActorOperationCapability, parameters: ActorOperationParameters, description: string): boolean {
  return capability.required_parameters.some((key) => {
    const value = key === "description" ? description : parameters[key as keyof ActorOperationParameters];
    return value === undefined || value === "" || typeof value === "string" && !value.trim();
  });
}
