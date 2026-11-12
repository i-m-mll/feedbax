import { relative, resolve } from 'node:path';
import ts from 'typescript';

const web = resolve(import.meta.dir, '..');
const src = resolve(web, 'src') + '/';
const configPath = resolve(web, 'tsconfig.json');
const config = ts.readConfigFile(configPath, ts.sys.readFile);
if (config.error) throw new Error(ts.flattenDiagnosticMessageText(config.error.messageText, '\n'));
const { fileNames, options } = ts.parseJsonConfigFileContent(config.config, ts.sys, web);
const modules = fileNames.filter((path) =>
  path.startsWith(src) && !path.startsWith(src + 'test/') && !/\.test\.[jt]sx?$|\.d\.ts$/.test(path),
);

const entries = [
  'main.tsx',
  'features/projectFolder/openProjectEditor.ts', // ed09a0a project-window mount
  // Reserved for gallery and statistics/replay surfaces.
  'components/panels/FigureGalleryPanel.tsx', // a5da2ff
  'components/panels/StatisticsPanel.tsx', // b2db05b
  'components/panels/TrajectoryViewer.tsx', // b2db05b
  // M2 view entry points. Mount when the project window and plan reader land.
  'components/scene/SceneRegion.tsx', // 1479f9c
  'components/scene/SceneClockControls.tsx', // 1479f9c
  'components/scene/useProjectPlanarScene.ts', // 78e8bea
  'components/pipeline/StudioPipelineMap.tsx', // 3e0ceba
  'components/pipeline/StudioRuns.tsx', // 339d687
  'components/pipeline/StudioRunInputs.tsx', // 339d687
  'components/pipeline/StudioDefinitionForm.tsx', // 339d687
  'components/pipeline/StudioSweepOverview.tsx', // 3a7b0c4
  // Owned by the legacy worker and controller removal issues.
  'hooks/useOrchestration.ts', // 2d1ed17
  'hooks/useTraining.ts', // 216f817
  'hooks/useWorkerConfig.ts', // 216f817
];
const program = ts.createProgram(fileNames, options);
const checker = program.getTypeChecker();
const resolutionCache = ts.createModuleResolutionCache(web, (path) => path, options);
const seen = new Set<ts.SourceFile>();

function visitModule(path: string) {
  const source = program.getSourceFile(path);
  if (!source) throw new Error(`No TypeScript source at ${relative(web, path)}`);
  if (seen.has(source)) return;
  seen.add(source);

  for (const { fileName } of ts.preProcessFile(source.text, true, true).importedFiles) {
    const specifier = fileName.replace(/\?sharedworker$/, '');
    const imported = ts.resolveModuleName(specifier, path, options, ts.sys, resolutionCache)
      .resolvedModule?.resolvedFileName;
    if (imported?.startsWith(src)) visitModule(imported);
  }
}

for (const entry of entries) visitModule(resolve(src, entry));
const unreachable = modules.filter((path) => {
  const source = program.getSourceFile(path);
  return !source || !seen.has(source);
});

const clients = new Map<ts.Symbol, ts.Declaration>();
for (const source of seen) {
  if (!source.fileName.startsWith(src + 'api/')) continue;
  const module = checker.getSymbolAtLocation(source);
  if (!module) continue;
  for (const symbol of checker.getExportsOfModule(module)) {
    const declaration = symbol.valueDeclaration;
    if (declaration && checker.getTypeOfSymbolAtLocation(symbol, declaration).getCallSignatures().length) {
      clients.set(symbol, declaration);
    }
  }
}

const referenced = new Set<ts.Symbol>();
function visitReferences(node: ts.Node) {
  if (ts.isImportDeclaration(node) || ts.isExportDeclaration(node)) return;
  if (ts.isIdentifier(node)) {
    const original = checker.getSymbolAtLocation(node);
    const symbol = original && (original.flags & ts.SymbolFlags.Alias)
      ? checker.getAliasedSymbol(original)
      : original;
    if (symbol && clients.has(symbol) && node.parent !== clients.get(symbol)) referenced.add(symbol);
  }
  ts.forEachChild(node, visitReferences);
}

for (const source of seen) visitReferences(source);
const uncalled = [...clients].filter(([symbol]) => !referenced.has(symbol));
for (const path of unreachable) console.error(`Unimported module: ${relative(web, path)}`);
for (const [symbol, declaration] of uncalled) {
  console.error(`Uncalled client: ${relative(web, declaration.getSourceFile().fileName)}#${symbol.name}`);
}
if (unreachable.length || uncalled.length) process.exitCode = 1;
