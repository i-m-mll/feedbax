import { Slider as BaseSlider } from '@base-ui/react/slider';
import clsx from 'clsx';
import { disabledLook } from './styles';

export interface SliderProps {
  value: number;
  min: number;
  max: number;
  step: number;
  onValueChange: (value: number) => void;
  'aria-label': string;
  disabled?: boolean;
  className?: string;
  /** `line` is a thin slider for a line of its own; `control` stands among controls. */
  size?: 'control' | 'line';
}

export function Slider({ value, min, max, step, onValueChange, disabled, className, size = 'control', 'aria-label': label }: SliderProps) {
  return (
    <BaseSlider.Root value={value} min={min} max={max} step={step} disabled={disabled}
      onValueChange={(next) => onValueChange(next)} className={clsx('w-full', disabledLook, className)}>
      <BaseSlider.Control className={clsx('flex w-full touch-none select-none items-center', size === 'line' ? 'h-4' : 'h-control')}>
        <BaseSlider.Track className="relative h-1 w-full rounded-full bg-shell-control">
          <BaseSlider.Indicator className="rounded-full bg-shell-foreground" />
          <BaseSlider.Thumb aria-label={label}
            className="size-3 rounded-full border border-shell-surface bg-shell-foreground has-[:focus-visible]:outline has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-offset-1 has-[:focus-visible]:outline-shell-accent" />
        </BaseSlider.Track>
      </BaseSlider.Control>
    </BaseSlider.Root>
  );
}
