import { readFileSync } from 'node:fs';
import { relative, resolve } from 'node:path';
import ts from 'typescript';
import vocabulary from '../src/components/activity/vocabulary.json';

const web = resolve(import.meta.dir, '..');
const src = resolve(web, 'src');
const sources = [
  ...new Bun.Glob('components/activity/*.tsx').scanSync({ cwd: src, absolute: true }),
  resolve(src, 'components/panels/StudioTrainPanel.tsx'),
];
const errors: string[] = [];
const retired = new RegExp(`\\b(?:${vocabulary.retired.join('|')})\\b`, 'i');

function verifyEntry(id: string, value: string) {
  if (retired.test(value)) errors.push(`Vocabulary ${id} uses a retired term`);
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

const controlledAttributes = new Set(['title', 'placeholder', 'aria-label']);
const labelFunctions = new Set(['studioName', 'studioAccessibleName']);
const visibleText = /[\p{L}\p{N}]/u;

function hasVisibleLiteral(node: ts.Expression): boolean {
  if (ts.isStringLiteralLike(node)) return visibleText.test(node.text);
  if (ts.isTemplateExpression(node)) {
    return visibleText.test(node.head.text) || node.templateSpans.some(
      (span) => visibleText.test(span.literal.text) || hasVisibleLiteral(span.expression),
    );
  }
  if (ts.isParenthesizedExpression(node)) return hasVisibleLiteral(node.expression);
  if (ts.isConditionalExpression(node)) {
    return hasVisibleLiteral(node.whenTrue) || hasVisibleLiteral(node.whenFalse);
  }
  if (ts.isBinaryExpression(node)) {
    const operator = node.operatorToken.kind;
    if (operator === ts.SyntaxKind.AmpersandAmpersandToken) return hasVisibleLiteral(node.right);
    if ([ts.SyntaxKind.BarBarToken, ts.SyntaxKind.QuestionQuestionToken, ts.SyntaxKind.PlusToken].includes(operator)) {
      return hasVisibleLiteral(node.left) || hasVisibleLiteral(node.right);
    }
  }
  return false;
}

for (const path of sources) {
  const source = ts.createSourceFile(path, readFileSync(path, 'utf8'), ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
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
    } else if (ts.isJsxAttribute(node) && controlledAttributes.has(node.name.getText(source))) {
      const value = node.initializer && ts.isJsxExpression(node.initializer) ? node.initializer.expression : undefined;
      if (!value || !ts.isCallExpression(value) || !ts.isIdentifier(value.expression) ||
        !labelFunctions.has(value.expression.text)) {
        report(node, `Use a Studio name for ${node.name.getText(source)}`);
      }
    }
    ts.forEachChild(node, visit);
  }
  visit(source);
}

for (const error of errors) console.error(error);
if (errors.length > 0) process.exitCode = 1;
