import { relative, resolve } from 'node:path';
import ts from 'typescript';
import vocabulary from '../src/components/activity/vocabulary.json';

const web = resolve(import.meta.dir, '..');
const src = resolve(web, 'src');
const laterMilestone = new Set(['useManagedAnalyze.tsx', 'useManagedReport.tsx', 'ManagedDocumentCanvas.tsx']);
const sources = [
  ...new Bun.Glob('components/activity/*.tsx').scanSync({ cwd: src, absolute: true }),
  ...new Bun.Glob('components/{pipeline,scene}/*.tsx').scanSync({ cwd: src, absolute: true }),
  ...new Bun.Glob('features/{projectWindow,projectFolder}/**/*.tsx').scanSync({ cwd: src, absolute: true })
    .filter((path) => !laterMilestone.has(path.split('/').at(-1)!)),
  resolve(src, 'App.tsx'),
  resolve(src, 'main.tsx'),
  resolve(src, 'components/layout/HeaderActivity.tsx'),
  resolve(src, 'components/layout/StudioErrorBoundary.tsx'),
  resolve(src, 'components/values/StudioFieldControl.tsx'),
].filter((path) => !path.endsWith('.test.tsx'));
const errors: string[] = [];
const retired = new RegExp(`\\b(?:${vocabulary.retired.join('|')})s?\\b`, 'i');

function verifyEntry(id: string, value: string) {
  if (retired.test(value.replace(/\{[A-Za-z]+\}/g, ''))) errors.push(`Vocabulary ${id} uses a retired term`);
}

for (const [id, value] of Object.entries(vocabulary.name)) {
  verifyEntry(id, value);
  const words = value.trim().split(/\s+/);
  if (!value.trim() || words.length > 2) errors.push(`Vocabulary ${id} must have one or two words`);
}
for (const [id, value] of Object.entries(vocabulary.reason)) {
  verifyEntry(id, value);
  if (!value.includes('{value}')) errors.push(`Vocabulary ${id} needs an object value`);
}
for (const group of Object.values(vocabulary.status)) {
  for (const [id, value] of Object.entries(group)) verifyEntry(id, value);
}
for (const [id, value] of Object.entries(vocabulary.format)) verifyEntry(id, value);

const names = new Map<string, readonly number[]>([
  ['studioName', [0]], ['studioAccessibleName', [0]], ['studioFieldLabel', [0]],
  ['studioFieldValueLabel', []], ['studioFieldKindLabel', [1]], ['studioReplayLabel', []],
]);
const vocabKeys = new Map<string, readonly number[]>([
  ...names, ['studioUnit', [0]], ['studioFormat', [0]], ['studioReason', [0]],
  ['studioDefinitionStatus', [0]], ['studioCheckpointStatus', [0]],
]);
const visibleText = /[\p{L}\p{N}]/u;
const textAttributes = new Set(['label', 'ariaLabel', 'aria-description']);

function branches(node: ts.Expression): readonly ts.Expression[] {
  if (ts.isParenthesizedExpression(node) || ts.isAsExpression(node) || ts.isSatisfiesExpression(node) ||
    ts.isNonNullExpression(node)) return [node.expression];
  if (ts.isConditionalExpression(node)) return [node.whenTrue, node.whenFalse];
  if (ts.isBinaryExpression(node)) {
    const operator = node.operatorToken.kind;
    if (operator === ts.SyntaxKind.AmpersandAmpersandToken) return [node.right];
    if ([ts.SyntaxKind.BarBarToken, ts.SyntaxKind.QuestionQuestionToken, ts.SyntaxKind.PlusToken].includes(operator)) {
      return [node.left, node.right];
    }
  }
  return [];
}

const options = ts.parseJsonConfigFileContent(
  ts.readConfigFile(resolve(web, 'tsconfig.json'), ts.sys.readFile).config, ts.sys, web,
).options;
const program = ts.createProgram(sources, options);
const checker = program.getTypeChecker();

function importedFunction(node: ts.Expression, module: RegExp) {
  if (!ts.isIdentifier(node)) return;
  const binding = checker.getSymbolAtLocation(node)?.declarations?.[0];
  if (!binding || !ts.isImportSpecifier(binding)) return;
  const imported = binding.parent.parent.parent.moduleSpecifier;
  if (ts.isStringLiteral(imported) && module.test(imported.text)) {
    return binding.propertyName?.text ?? binding.name.text;
  }
}

function vocabularyFunction(node: ts.Expression) {
  return importedFunction(node, /(?:^|\/)studioVocabulary$/);
}

function renderedInitializer(node: ts.Identifier): ts.Expression | undefined {
  const binding = checker.getSymbolAtLocation(node)?.valueDeclaration;
  return binding && ts.isVariableDeclaration(binding) && ts.isVariableDeclarationList(binding.parent) &&
    (binding.parent.flags & ts.NodeFlags.Const) !== 0 ? binding.initializer : undefined;
}

function mappedInput(node: ts.Identifier): ts.Expression | undefined {
  const declaration = checker.getSymbolAtLocation(node)?.valueDeclaration;
  if (!declaration || !ts.isParameter(declaration) || !ts.isArrowFunction(declaration.parent)) return;
  const callback = declaration.parent;
  if (callback.parameters[0] !== declaration || !ts.isCallExpression(callback.parent)) return;
  const call = callback.parent;
  if (call.arguments[0] === callback && ts.isPropertyAccessExpression(call.expression) && call.expression.name.text === 'map') {
    return call.expression.expression;
  }
}

function hasVisibleLiteral(node: ts.Expression, visited = new Set<ts.Expression>()): boolean {
  if (visited.has(node)) return false;
  visited.add(node);
  const visible = (expression: ts.Expression) => hasVisibleLiteral(expression, visited);
  if (ts.isStringLiteralLike(node)) return visibleText.test(node.text);
  if (ts.isTemplateExpression(node)) return visibleText.test(node.head.text) ||
    node.templateSpans.some((span) => visibleText.test(span.literal.text) || visible(span.expression));
  if (ts.isArrayLiteralExpression(node)) return node.elements.some(visible);
  if (ts.isObjectLiteralExpression(node)) return node.properties.some((part) =>
    ts.isPropertyAssignment(part) && visible(part.initializer));
  if (ts.isIdentifier(node)) {
    const initializer = renderedInitializer(node) ?? mappedInput(node);
    return initializer !== undefined && visible(initializer);
  }
  if (ts.isCallExpression(node)) {
    const name = vocabularyFunction(node.expression);
    const keys = name && vocabKeys.get(name);
    if (keys) return node.arguments.some((argument, index) => !keys.includes(index) && visible(argument));
    if (ts.isPropertyAccessExpression(node.expression) && ['join', 'toLowerCase', 'toUpperCase'].includes(node.expression.name.text)) {
      return visible(node.expression.expression) || node.arguments.some(visible);
    }
  }
  return branches(node).some(visible);
}

function isName(node: ts.Expression): boolean {
  if (ts.isIdentifier(node)) return node.text === 'undefined';
  if (ts.isCallExpression(node)) return names.has(vocabularyFunction(node.expression) ?? '') && !hasVisibleLiteral(node);
  const parts = branches(node);
  return parts.length > 0 && (!ts.isBinaryExpression(node) || node.operatorToken.kind !== ts.SyntaxKind.PlusToken) &&
    parts.every(isName);
}

for (const path of sources) {
  const source = program.getSourceFile(path);
  if (!source) throw new Error(`Studio vocabulary source missing: ${relative(web, path)}`);
  const report = (node: ts.Node, message: string) => {
    const { line } = source.getLineAndCharacterOfPosition(node.getStart(source));
    errors.push(`${relative(web, path)}:${line + 1} ${message}`);
  };
  function visit(node: ts.Node) {
    if (ts.isJsxText(node) && visibleText.test(node.text)) {
      report(node, 'Use a Studio vocabulary entry for visible text');
    } else if (ts.isJsxExpression(node) && node.expression &&
      (ts.isJsxElement(node.parent) || ts.isJsxFragment(node.parent)) &&
      hasVisibleLiteral(node.expression)) {
      report(node, 'Use a Studio vocabulary entry for visible text');
    } else if (ts.isJsxAttribute(node)) {
      const attribute = node.name.getText(source);
      const value = node.initializer && ts.isJsxExpression(node.initializer) ? node.initializer.expression : undefined;
      if (attribute === 'title') report(node, 'Use the shell tooltip or a visible label');
      if ((attribute === 'placeholder' || attribute === 'aria-label') && (!value || !isName(value))) {
        report(node, `Use a Studio name for ${attribute}`);
      }
      if (textAttributes.has(attribute)) {
        if ((node.initializer && ts.isStringLiteral(node.initializer) && visibleText.test(node.initializer.text)) ||
          (value && hasVisibleLiteral(value))) report(node, `Use Studio vocabulary for ${attribute}`);
      }
    } else if (ts.isCallExpression(node) &&
      importedFunction(node.expression, /^@window-shell\/browser\/tooltip$/) === 'appTip' &&
      node.arguments[0] && !isName(node.arguments[0])) {
      report(node, 'Use a Studio name for the shell tooltip');
    }
    ts.forEachChild(node, visit);
  }
  visit(source);
}

for (const error of errors) console.error(error);
if (errors.length > 0) process.exitCode = 1;
