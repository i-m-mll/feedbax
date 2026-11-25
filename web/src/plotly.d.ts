/** Minimal type declarations for plotly.js-dist-min. */
declare module 'plotly.js-dist-min' {
  export interface Data {
    [key: string]: unknown;
  }

  export interface Layout {
    [key: string]: unknown;
    autosize?: boolean;
    margin?: { t?: number; r?: number; b?: number; l?: number };
  }

  export interface Config {
    responsive?: boolean;
    displayModeBar?: boolean;
    [key: string]: unknown;
  }

  export function newPlot(
    root: HTMLElement,
    data: Data[],
    layout?: Partial<Layout>,
    config?: Config,
  ): Promise<HTMLElement>;

  export function newPlot(
    root: HTMLElement,
    figure: { data: Data[]; layout?: Partial<Layout>; frames?: Record<string, unknown>[]; config?: Config },
  ): Promise<HTMLElement>;

  export function purge(root: HTMLElement): void;

  export function restyle(root: HTMLElement, update: Record<string, unknown[]>, traces: number[]): Promise<HTMLElement>;

  /** A drawn axis: its pixel offset and length in the plot, and its maps between data values and pixels from its offset. */
  export interface FullAxis {
    _offset: number;
    _length: number;
    l2p(value: number): number;
    p2l(pixel: number): number;
  }

  export interface ClickPoint {
    curveNumber: number;
    y: unknown;
  }

  /** The plot element after `newPlot`, with the layout Plotly drew and its events. */
  export interface PlotElement extends HTMLElement {
    _fullLayout: Record<string, unknown>;
    on(event: 'plotly_click', handler: (event: { points: ClickPoint[]; event: MouseEvent }) => void): void;
    on(event: 'plotly_afterplot' | 'plotly_relayout', handler: () => void): void;
  }
}
