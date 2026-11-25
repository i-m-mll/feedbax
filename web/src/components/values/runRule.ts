/** What a run rule can name: the axes of the runs it reads, and the final metrics it compares or ranks by. */
export interface RuleVocabulary {
  /** Each axis, with its values as a rule writes them (`0`, `fast`, `"a b"`). */
  readonly axes: Readonly<Record<string, readonly string[]>>;
  /** The final metrics the producer's training method declares, where the rule may read results; none where it is
   * resolved before its runs have results. */
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
/** A final metric is a number, which a rule orders against a threshold. */
const metricOperators = ['<', '<=', '>', '>='] as const;
const named = String.raw`([a-zA-Z_][\w-]*|"[^"]*")`;
const comparison = new RegExp(`^${named}(==|!=|<=|>=|=|<|>)(.*)$`, 's');
const partialOperator = new RegExp(`^${named}([!<>=]+)$`);
/** The name a word writes: its text, or the text inside its quotes. */
const spelled = (word: string) => word.startsWith('"') ? word.slice(1, -1) : word;

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
 * What can follow the caret in `rule`: an axis name, then its operators, then its values; where the rule may read
 * results, a metric name, then the operators that order it, and `lowest` and `highest`, with a metric name after them,
 * then `per` and axes.
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
  const isAxis = (name: string) => Object.prototype.hasOwnProperty.call(axes, name);
  const isMetric = (name: string) => !isAxis(name) && !!metrics?.includes(name);
  const compared = comparison.exec(word);
  if (compared && isAxis(compared[1])) {
    const [, name, operator, rest] = compared;
    const fragment = operator === '=' ? rest.split(/,|\.\./).at(-1)! : rest;
    return offer(fragment, axes[name]);
  }
  const partial = partialOperator.exec(word);
  if (partial && isAxis(partial[1])) return offer(partial[2], operators);
  if (partial && isMetric(spelled(partial[1]))) return offer(partial[2], metricOperators);
  if (compared && isMetric(spelled(compared[1]))) return [];
  const follow = (choices: readonly string[]) => {
    const start = before.length;
    return choices.map((operator) => ({ text: before + operator + after, caret: start + operator.length, label: operator }));
  };
  if (isAxis(word)) return follow(operators);
  if (isMetric(spelled(word)) && word === ruleWord(spelled(word))) return follow(metricOperators);
  if (word && !/^(?:[a-zA-Z_][\w-]*|"[^"]*)$/.test(word)) return [];
  return [...offer(word, names), ...metrics ? [...offer(word, metrics, ruleWord), ...offer(word, [...ranks])] : []];
}
