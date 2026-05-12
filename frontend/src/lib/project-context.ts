import projectContextJson from "../data/generated/project-context.json";
import type { ProjectContext, UserInputFieldName } from "../types";

export const projectContext = projectContextJson as ProjectContext;

export const orderedFieldNames = Object.keys(projectContext.schema) as UserInputFieldName[];

export function getActionSpec(actionKey: string) {
  return projectContext.policy.actionCatalog.find((item) => item.action_key === actionKey);
}
