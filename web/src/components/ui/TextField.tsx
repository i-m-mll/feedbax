import { Input } from '@base-ui/react/input';
import clsx from 'clsx';
import { appTip } from '@window-shell/browser/tooltip';
import {
  forwardRef, useCallback, useImperativeHandle, useLayoutEffect, useRef, useState, type ComponentPropsWithoutRef, type CSSProperties,
  type ReactNode, type TextareaHTMLAttributes,
} from 'react';
import { fieldBox, fieldFrame, fieldInput, glyph, leadingSlot } from './styles';

export interface TextFieldProps extends Omit<ComponentPropsWithoutRef<typeof Input>, 'className' | 'onValueChange'> {
  onValueChange?: (value: string) => void;
  invalid?: boolean;
  leading?: ReactNode;
  trailing?: ReactNode;
  /** A control sharing the box, at its end past a hairline: a class's picker. */
  end?: ReactNode;
  /** A term of the value drawn in the problem colour, its tip the reason. */
  term?: TextTerm;
  /** Muted text drawn right after the value: a time's value in the other unit. */
  after?: { readonly text: string; readonly id: string };
  className?: string;
}

/** The characters `start` to `end` of a field's text, and why they are wrong. */
export interface TextTerm {
  readonly start: number;
  readonly end: number;
  readonly reason: string;
}

/**
 * The input's text drawn again over it, glyph for glyph: with a term, the term in the problem colour and the input's own
 * glyphs transparent beneath; with `after`, that text following the value. The term takes the pointer for its tip and
 * gives the keyboard to the input at its start.
 */
export function TermText({ input, text, term, after }: {
  input: HTMLInputElement; text: string; term?: TextTerm; after?: TextFieldProps['after'];
}) {
  const [style, setStyle] = useState<CSSProperties>();
  const layer = useRef<HTMLSpanElement>(null);
  useLayoutEffect(() => {
    const computed = getComputedStyle(input);
    setStyle({
      left: input.offsetLeft, top: input.offsetTop, width: input.offsetWidth, height: input.offsetHeight,
      font: computed.font, letterSpacing: computed.letterSpacing, paddingLeft: computed.paddingLeft,
      paddingRight: computed.paddingRight,
    });
    const follow = () => { if (layer.current) layer.current.scrollLeft = input.scrollLeft; };
    follow();
    input.addEventListener('scroll', follow);
    input.addEventListener('input', follow);
    return () => { input.removeEventListener('scroll', follow); input.removeEventListener('input', follow); };
  }, [input, text]);
  if (!style) return null;
  return <span ref={layer} aria-hidden="true" style={style}
    className="pointer-events-none absolute flex items-center overflow-hidden whitespace-pre">
    <span className={term ? 'text-shell-foreground' : 'text-transparent'}>
      {term ? <>
        {text.slice(0, term.start)}
        <span data-field-term="" {...appTip(term.reason)} className="pointer-events-auto text-shell-status-failure"
          onMouseDown={(event) => {
            event.preventDefault();
            input.focus();
            input.setSelectionRange(term.start, term.start);
          }}>{text.slice(term.start, term.end)}</span>
        {text.slice(term.end)}
      </> : text}
    </span>
    {after && <span data-field-after="" className="text-shell-muted">{' · '}<span id={after.id}>{after.text}</span></span>}
  </span>;
}

export const TextField = forwardRef<HTMLInputElement, TextFieldProps>(function TextField(
  { onValueChange, invalid, leading, trailing, end, term, after, className, ...props }, ref,
) {
  const [input, setInput] = useState<HTMLInputElement | null>(null);
  useImperativeHandle(ref, () => input!, [input]);
  const attach = useCallback((node: HTMLElement | null) => setInput(node as HTMLInputElement | null), []);
  const text = typeof props.value === 'string' ? props.value : '';
  const shown = term && input && term.end <= text.length ? term : undefined;
  const layered = !!input && (!!shown || !!after);
  return (
    <div data-field-box="" className={clsx(fieldBox, layered && 'relative', className)}>
      {leading ? <span className={leadingSlot}>{leading}</span> : null}
      <Input
        ref={attach}
        aria-invalid={invalid || undefined}
        className={clsx(fieldInput, leading && 'pl-1', shown && 'text-transparent caret-shell-foreground')}
        onValueChange={onValueChange && ((value) => onValueChange(value))}
        {...props}
      />
      {layered && <TermText input={input!} text={text} term={shown} after={after} />}
      {trailing ? <span className={clsx('flex pr-control-inline text-shell-muted', glyph)}>{trailing}</span> : null}
      {end ? <span className="field-end flex h-full min-w-control shrink-[1000] border-l border-shell-field">{end}</span> : null}
    </div>
  );
});

export interface TextAreaProps extends Omit<TextareaHTMLAttributes<HTMLTextAreaElement>, 'className'> {
  invalid?: boolean;
  className?: string;
}

export const TextArea = forwardRef<HTMLTextAreaElement, TextAreaProps>(function TextArea(
  { invalid, className, ...props }, ref,
) {
  return (
    <div className={clsx(fieldFrame, className)}>
      <textarea
        ref={ref}
        aria-invalid={invalid || undefined}
        className={clsx(fieldInput, 'min-h-[calc(3*var(--row-height))] resize-y py-1')}
        {...props}
      />
    </div>
  );
});
