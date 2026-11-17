import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { relative, resolve } from 'node:path';
import ts from 'typescript';

const web = resolve(import.meta.dir, '..');
const src = resolve(web, 'src');
const tokenFiles = new Set([resolve(src, 'index.css')]);
const controlLayer = resolve(src, 'components/ui') + '/';
const rawControls = new Set(['button', 'input', 'select', 'textarea']);

const colors = createRequire(import.meta.url)('tailwindcss/colors');
const palette = Object.entries(Object.getOwnPropertyDescriptors(colors))
  .filter(([, { value }]) => typeof value === 'object' || typeof value === 'string' && value.startsWith('#'))
  .map(([name]) => name);
const utilities = 'bg|text|border(?:-[xytrblse])?|ring(?:-offset)?|outline|divide|fill|stroke|from|via|to|shadow|accent|caret|decoration|placeholder';
const paletteClass = new RegExp(`(?<![\\w-])-?(?:${utilities})-(?:${palette.join('|')})(?:-\\d{2,3})?(?:/\\d+)?(?![\\w-])`);
const colourLiteral = /(?<![\w&/-])#(?:[0-9a-fA-F]{3,4}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})(?![\w-])|(?<![\w-])(?:rgba?|hsla?|hwb|lab|lch|oklab|oklch|color)\(/;
const cssNamedColour = /(?<![\w-])(?:white|black)(?![\w-])/;

const errors: string[] = [];
const report = (path: string, line: number, message: string) => errors.push(`${relative(web, path)}:${line} ${message}`);

function checkText(path: string, line: number, text: string) {
  const palettes = text.match(paletteClass);
  if (palettes) report(path, line, `Tailwind palette class ${palettes[0].trim()}; use a shell-* token class`);
  const literal = text.match(colourLiteral);
  if (literal) report(path, line, `colour literal ${literal[0]}; define a token in src/index.css`);
}

function checkScript(path: string) {
  const source = ts.createSourceFile(path, readFileSync(path, 'utf8'), ts.ScriptTarget.Latest, true);
  const line = (node: ts.Node) => source.getLineAndCharacterOfPosition(node.getStart(source)).line + 1;
  const visit = (node: ts.Node) => {
    if ((ts.isJsxOpeningElement(node) || ts.isJsxSelfClosingElement(node)) && ts.isIdentifier(node.tagName) &&
      rawControls.has(node.tagName.text) && !path.startsWith(controlLayer)) {
      report(path, line(node), `raw <${node.tagName.text}>; use the control from components/ui`);
    }
    if (ts.isStringLiteral(node) || ts.isNoSubstitutionTemplateLiteral(node) || ts.isTemplateHead(node) ||
      ts.isTemplateMiddle(node) || ts.isTemplateTail(node)) {
      checkText(path, line(node), node.text);
    }
    ts.forEachChild(node, visit);
  };
  visit(source);
}

function checkStyles(path: string) {
  const text = readFileSync(path, 'utf8').replace(/\/\*[\s\S]*?\*\//g, (comment) => comment.replace(/[^\n]/g, ' '));
  text.split('\n').forEach((content, index) => {
    const value = content.match(/:\s*([^;{}]*)/)?.[1] ?? '';
    const literal = value.match(colourLiteral) ?? value.match(cssNamedColour);
    if (literal) report(path, index + 1, `colour literal ${literal[0]}; define a token in src/index.css`);
  });
}

const sources = (pattern: string) => [...new Bun.Glob(pattern).scanSync({ cwd: src, absolute: true })];
for (const path of sources('**/*.{ts,tsx}')) {
  if (!/\.test\.tsx?$|\.d\.ts$/.test(path)) checkScript(path);
}
for (const path of sources('**/*.css')) {
  if (!tokenFiles.has(path)) checkStyles(path);
}
checkScript(resolve(web, 'tailwind.config.js'));

if (errors.length) {
  console.error(errors.join('\n'));
  console.error(`\n${errors.length} control-layer violation${errors.length === 1 ? '' : 's'}`);
  process.exit(1);
}
