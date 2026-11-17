export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    fontSize: {
      xs: 'var(--text-xs)',
      sm: 'var(--text-sm)',
      base: 'var(--text-md)',
      lg: 'var(--text-lg)',
      xl: 'var(--text-xl)'
    },
    extend: {
      borderColor: {
        DEFAULT: 'var(--app-divider)'
      },
      colors: {
        shell: {
          chrome: 'var(--surface-header)',
          surface: 'var(--surface-main)',
          subtle: 'var(--surface-toolbar)',
          divider: 'var(--app-divider)',
          control: 'var(--studio-control-border)',
          foreground: 'var(--text-primary)',
          muted: 'var(--text-muted)',
          hover: 'var(--surface-hover)',
          'row-hover': 'var(--row-hover-fill)',
          'row-selected': 'var(--row-selected-fill)',
          'row-related': 'var(--row-related-mark)',
          accent: 'var(--studio-shell-accent)',
          selected: 'var(--studio-shell-selection)',
          scrim: 'var(--studio-scrim)',
          'status-idle': 'var(--studio-status-idle)',
          'status-neutral': 'var(--studio-status-neutral)',
          'status-warning': 'var(--studio-status-warning)',
          'status-running': 'var(--studio-status-running)',
          'status-success': 'var(--studio-status-success)',
          'status-failure': 'var(--studio-status-failure)',
          'status-ink': 'var(--studio-status-ink)',
          'scene-feature': 'var(--studio-scene-feature)',
          'scene-guide': 'var(--studio-scene-guide)',
        }
      },
      spacing: {
        control: 'var(--control-height)',
        'control-inline': 'var(--control-padding-inline)',
        row: 'var(--row-height)',
        'row-inline': 'var(--row-padding-inline)',
        'panel-header': 'var(--panel-header-height)',
        'panel-header-inline': 'var(--panel-header-padding-inline)',
        'shell-inset': 'var(--space-3)'
      },
      borderRadius: {
        'shell-control': 'var(--radius-lg)',
        control: 'var(--radius-md)',
        row: 'var(--radius-sm)',
        surface: 'var(--radius-lg)'
      },
      boxShadow: {
        soft: 'var(--studio-shadow-soft)',
        lift: 'var(--studio-shadow-lift)'
      },
      fontFamily: {
        sans: 'var(--font-ui)'
      }
    }
  },
  plugins: []
};
