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
          accent: 'var(--studio-shell-accent)',
          selected: 'var(--studio-shell-selection)',
          'status-idle': 'var(--studio-status-idle)',
          'status-neutral': 'var(--studio-status-neutral)',
          'status-warning': 'var(--studio-status-warning)',
          'status-running': 'var(--studio-status-running)',
          'status-success': 'var(--studio-status-success)',
          'status-failure': 'var(--studio-status-failure)',
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
        'shell-control': 'var(--radius-lg)'
      },
      boxShadow: {
        soft: '0 12px 32px rgba(15, 23, 42, 0.08)',
        lift: '0 18px 40px rgba(15, 23, 42, 0.12)'
      },
      fontFamily: {
        sans: 'var(--font-ui)'
      }
    }
  },
  plugins: []
};
