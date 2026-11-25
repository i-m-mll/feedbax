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
