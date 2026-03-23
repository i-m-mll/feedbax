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

  export function purge(root: HTMLElement): void;
}
