import path from 'node:path';
import type { PluginObj, types as BabelTypes } from '@babel/core';

const repository = path.resolve(__dirname, '..');
const sources = path.join(__dirname, 'src') + path.sep;

/**
 * Stamps each intrinsic JSX element written in `web/src` with `data-source="web/src/…tsx:line"`, the place it is written,
 * which the notes toolbar reads when React gives no source.
 */
export function sourceStamps({ types: t }: { types: typeof BabelTypes }): PluginObj {
  return {
    name: 'feedbax-source-stamps',
    visitor: {
      JSXOpeningElement(element, state) {
        const { name, attributes, loc } = element.node;
        const file = state.filename;
        if (!loc || !file?.startsWith(sources) || !t.isJSXIdentifier(name) || !/^[a-z]/.test(name.name)) return;
        if (attributes.some((attribute) => t.isJSXAttribute(attribute) && attribute.name.name === 'data-source')) return;
        const source = `${path.relative(repository, file).split(path.sep).join('/')}:${loc.start.line}`;
        attributes.push(t.jsxAttribute(t.jsxIdentifier('data-source'), t.stringLiteral(source)));
      },
    },
  };
}
