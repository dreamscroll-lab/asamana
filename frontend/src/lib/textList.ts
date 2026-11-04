/**
 * A short list of phrases (traits, values, relation labels) edited as one line of text. 「、」,
 * 「，」 and "," all separate; blank items are dropped.
 */
export function splitList(text: string): string[] {
  return text.split(/[、，,]/).map((item) => item.trim()).filter(Boolean);
}

export function joinList(items: readonly string[]): string {
  return items.join("、");
}
