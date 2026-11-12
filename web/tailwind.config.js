export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        ink: '#0b0d10',
        shell: {
          chrome: 'var(--surface-header)',
          divider: 'var(--app-divider)',
          foreground: 'var(--text-primary)',
          muted: 'var(--text-muted)',
          hover: 'var(--surface-hover)',
          accent: 'var(--studio-shell-accent)',
          selected: 'var(--studio-shell-selection)'
        },
        slate: {
          50: '#f7f7f8',
          100: '#ededf0',
          200: '#d9d9df',
          300: '#bfc0c8',
          400: '#9b9cab',
          500: '#7d7f92',
          600: '#5f6176',
          700: '#44465a',
          800: '#2d2f3f',
          900: '#1f202c'
        },
        brand: {
          500: '#2f7cf6',
          600: '#2263d6'
        },
        mint: {
          500: '#2fbf7f'
        },
        amber: {
          500: '#f2b92d'
        }
      },
      spacing: {
        'shell-header': 'var(--app-header-height)',
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
        display: ['"Space Grotesk"', '"Segoe UI"', 'sans-serif'],
        sans: ['"Work Sans"', '"Segoe UI"', 'sans-serif']
      }
    }
  },
  plugins: []
};
