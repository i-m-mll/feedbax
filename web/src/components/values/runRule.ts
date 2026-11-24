/** What a run rule can name: the axes of the runs it reads, and the results a lowest or highest term ranks by. */
export interface RuleVocabulary {
  /** Each axis, with its values as a rule writes them (`0`, `fast`, `"a b"`). */
  readonly axes: Readonly<Record<string, readonly string[]>>;
  /** The result names, where the rule may rank; none where it is resolved before its runs have results. */
  readonly metrics?: readonly string[];
}

/** One completion: the rule with it taken, the caret after it, and what the list shows. */
export interface RuleCompletion {
  readonly text: string;
  readonly caret: number;
  readonly label: string;
}

const ruleName = /^[a-zA-Z_](?:[a-zA-Z0-9_-]*[a-zA-Z0-9_])?$/;
const reserved = new Set(['true', 'false', 'and', 'or', 'not', 'lowest', 'highest', 'per']);
const ranks = new Set(['lowest', 'highest']);
const operators = ['=', '!=', '>=', '<=', '>', '<'] as const;
const comparison = /^([a-zA-Z_][\w-]*)(==|!=|<=|>=|=|<|>)(.*)$/s;
const partialOperator = /^([a-zA-Z_][\w-]*)([!<>=]+)$/;

/** A name as a rule writes it: bare when it is a rule name, else quoted. */
export function ruleWord(name: string) {
  return ruleName.test(name) && !reserved.has(name) ? name : JSON.stringify(name);
}

/**
 * The rule choosing, on each axis, the values whose ids are listed, in the axis's order; `values[axis][id]` is how a
 * rule writes each.
 */
export function valuesRule(values: Readonly<Record<string, Readonly<Record<string, string>>>>,
  chosen: Readonly<Record<string, readonly string[]>>) {
  return Object.keys(chosen).sort().map((axis) => {
    const ids = new Set(chosen[axis]);
    const known = Object.keys(values[axis] ?? {});
    const ordered = [...known.filter((id) => ids.has(id)), ...[...ids].filter((id) => !known.includes(id)).sort()];
    return `${axis}=${ordered.map((id) => values[axis]?.[id] ?? ruleWord(id)).join(',')}`;
  }).join(' ');
}

/** A run as a rule reads it: its id, and its value's id on each axis. */
export interface RuledRun {
  readonly id: string;
  readonly values: Readonly<Record<string, string>>;
}

/**
 * The shortest rule choosing exactly the `picked` runs of `runs`: a term for each axis on which the picked runs leave
 * a value out, and none where they are every run. Undefined when no rule of `axes` terms chooses exactly them.
 */
export function pickedRule(runs: readonly RuledRun[], picked: ReadonlySet<string>,
  axes: Readonly<Record<string, Readonly<Record<string, string>>>>): string | undefined {
  const chosen = runs.filter(({ id }) => picked.has(id));
  if (!chosen.length) return undefined;
  const terms: Record<string, string[]> = {};
  for (const axis of Object.keys(axes)) {
    const taken = new Set(chosen.map(({ values }) => values[axis]));
    if (taken.size === new Set(runs.map(({ values }) => values[axis])).size) continue;
    const ids = [...taken].filter((id): id is string => id !== undefined && id in axes[axis]);
    if (ids.length < taken.size) return undefined;
    terms[axis] = ids;
  }
  const matched = runs.filter(({ values }) => Object.entries(terms).every(([axis, ids]) => ids.includes(values[axis])));
  return matched.length === chosen.length ? valuesRule(axes, terms) : undefined;
}

/** What picked runs write: the shortest rule choosing them, else their rows pinned as To Rows pins them. */
export type PickedTerms<Row> = { readonly rule: string } | { readonly rows: readonly Row[] };

/** The terms that choose exactly the `picked` runs; `rowOf` is the pinned row of each run. */
export function pickedTerms<Row>(runs: readonly RuledRun[], picked: ReadonlySet<string>,
  axes: Readonly<Record<string, Readonly<Record<string, string>>>>, rowOf: (run: string) => Row | undefined): PickedTerms<Row> {
  const rule = pickedRule(runs, picked, axes);
  if (rule !== undefined) return { rule };
  return { rows: runs.flatMap(({ id }) => {
    const row = picked.has(id) ? rowOf(id) : undefined;
    return row === undefined ? [] : [row];
  }) };
}

/** The rule's words before the caret; a quoted name is one word, spaces and all. */
function words(text: string): string[] {
  const found: string[] = [];
  let word = '';
  let quoted = false;
  for (const char of text) {
    if (char === '"') quoted = !quoted;
    if (!quoted && /\s/.test(char)) {
      if (word) found.push(word);
      word = '';
    } else word += char;
  }
  return [...found, word];
}

/**
 * What can follow the caret in `rule`: an axis name, then its operators, then its values; `lowest` and `highest`
 * where the rule may rank, and after them a result name, then `per` and axes. A result name never starts a comparison.
 */
export function ruleCompletions(rule: string, caret: number, { axes, metrics }: RuleVocabulary): RuleCompletion[] {
  const before = rule.slice(0, caret);
  const after = rule.slice(caret);
  const all = words(before);
  const word = all.at(-1)!;
  const previous = all.slice(0, -1);
  /** The choices that extend `fragment`, the word's end the caret is in; `insert` is how the rule writes each. */
  const offer = (fragment: string, choices: readonly string[], insert: (choice: string) => string = (choice) => choice) => {
    const start = before.length - fragment.length;
    const typed = fragment.replace(/^"/, '').toLowerCase();
    return choices.filter((choice) => choice.toLowerCase() !== typed && choice.toLowerCase().startsWith(typed))
      .map((choice) => ({ text: before.slice(0, start) + insert(choice) + after, caret: start + insert(choice).length, label: choice }));
  };
  const names = Object.keys(axes);
  const rank = previous.map((item) => ranks.has(item)).lastIndexOf(true);
  if (rank >= 0 && metrics) {
    const [count, metric, per, ...rest] = previous.slice(rank + 1);
    if (count === undefined) return [];
    if (metric === undefined) {
      return offer(word, metrics, ruleWord);
    }
    if (per === 'per' && !rest.length) return offer(word.split(',').at(-1)!, names);
    if (per === undefined && 'per'.startsWith(word) && word) return offer(word, ['per']);
  }
  const compared = comparison.exec(word);
  if (compared && compared[1] in axes) {
    const [, name, operator, rest] = compared;
    const fragment = operator === '=' ? rest.split(/,|\.\./).at(-1)! : rest;
    return offer(fragment, axes[name]);
  }
  const partial = partialOperator.exec(word);
  if (partial && partial[1] in axes) return offer(partial[2], operators);
  if (word in axes) {
    const start = before.length;
    return operators.map((operator) => ({ text: before + operator + after, caret: start + operator.length, label: operator }));
  }
  if (word && !/^[a-zA-Z_][\w-]*$/.test(word)) return [];
  return offer(word, [...names, ...metrics ? [...ranks] : []]);
}
