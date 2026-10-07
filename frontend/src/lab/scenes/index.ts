/**
 * The scene bench's running order. Scenes live one file per sidebar group; payload shapes and the
 * wire-fidelity note live in `fixtures.ts`; scenes are built from roles that `places.ts` casts from
 * the loaded map.
 */

import type { LabPlaces } from "../places";
import { type LabScene, resetFixtures } from "./fixtures";
import { buildMovementScenes } from "./movement";
import { buildPhysicalScenes } from "./physical";
import { buildSpeechScenes } from "./speech";
import { buildCovertScenes } from "./covert";
import { buildOngoingScenes } from "./ongoing";
import { buildMadeScenes } from "./made";
import { buildPocketScenes } from "./pocket";
import { buildErrandScenes } from "./errand";
import { buildWorldScenes } from "./world";
import { buildStagingScenes } from "./staging";
import { buildBodyScenes } from "./bodies";
import { buildWeatherScenes } from "./weather";

export { CAST, castDemographics } from "./fixtures";
export type { LabActor, LabScene } from "./fixtures";

/**
 * Every scene, built for this map's places (see places.ts). The order is the sidebar's and also the
 * order the fixture counters number steps in, so `resetFixtures` must run here first.
 */
export function buildScenes(places: LabPlaces): LabScene[] {
  resetFixtures();
  return [
    ...buildMovementScenes(places),
    ...buildPhysicalScenes(places),
    ...buildSpeechScenes(places),
    ...buildCovertScenes(places),
    ...buildOngoingScenes(places),
    ...buildPocketScenes(places),
    ...buildErrandScenes(places),
    ...buildMadeScenes(places),
    ...buildWorldScenes(places),
    ...buildStagingScenes(places),
    ...buildBodyScenes(places),
    ...buildWeatherScenes(places),
  ];
}
