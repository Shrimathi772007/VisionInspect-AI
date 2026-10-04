import { apiFetch } from "./client";

/** GET /ai/models - the served model of every MVTec category with its gate and static final-test metrics. */
export function getAiModels() {
  return apiFetch("/ai/models");
}
