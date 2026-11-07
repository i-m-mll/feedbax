import { relative, resolve } from "node:path";
import { createVitest } from "vitest/node";

const vitest = await createVitest("test", {
  watch: false,
  related: process.argv.slice(2),
});
try {
  const root = resolve("..");
  const specs = await vitest.getRelevantTestSpecifications();
  console.log(JSON.stringify(specs.map((spec) => relative(root, spec.moduleId))));
} finally {
  await vitest.close();
}
