# Figure Review Dashboard

## Overview

Interactive web dashboard built with Dash for reviewing and organizing figures generated during model evaluations. Provides a grid-based interface for viewing figures filtered by database columns and arranged by hyperparameter values.

## Architecture

### Backend Components

**FigureQueryEngine** (`backend/query.py`)
- Executes JOIN queries across `FigureRecord` and `EvaluationRecord` tables
- Applies user-specified filters and returns filtered DataFrames with prefixed column names (`table.column`)
- Identifies "distinguishing columns" (columns with >1 unique value in filtered results)
- Validates grid uniqueness (ensures row/col combination uniquely identifies records)
- Builds grid data structures mapping (row_val, col_val) tuples to figure records

**FigureLoader** (`backend/loader.py`)
- Loads figure files from disk and converts to base64 data URIs for browser display
- Supports multiple formats (PNG, SVG, JSON for Plotly figures)

**PresetManager** (`backend/presets.py`)
- Persists filter configurations as JSON files in `{PATHS.db}/dashboard_presets/`
- Supports save, load, delete, and list operations

### Database Schema

Queries join two tables:
- `FigureRecord`: figure metadata (identifier, figure_type, saved_formats, evaluation_hash)
- `EvaluationRecord`: evaluation parameters and model metadata

Both tables store hyperparameter values as columns (e.g., `pert__std`, `pert__type`). The query engine returns a combined DataFrame with prefixed columns to disambiguate overlapping names.

## UI Flow

### 1. Figure Type Filter
- Locked row pinned at the top of the filter builder
- Table/Column/Operator locked to `figures.identifier =`
- Value dropdown lists every identifier present in the database (highlighted until chosen)

### 2. Filter Management
- Card-based filter builder with one row per condition (dropdowns for Table/Column/Operator/Value)
- Value field is a dropdown filled with distinct values from the selected column (searchable, limited to existing DB entries)
- Add/remove rows inline; rows validate against the selected table's columns
- Filters sync to a Dash Store and trigger DataFrame updates
- Supports standard SQL-style operators (=, !=, <, >, <=, >=, IN, LIKE, IS NULL, IS NOT NULL)

### 3. Filter Presets
- Save current filter configuration with a custom name
- Load previously saved configurations
- Delete unwanted presets
- Presets persist across sessions

### 4. Distinguishing Columns Display
- Shows all columns that still vary in the filtered dataset
- Displays: Table name (separate column), Column name, Count of unique values, Sample values
- Excludes internal columns (id, hash, created_at, archived fields)
- Used to guide grid dimension selection

### 5. Grid Configuration
- Select row dimension (optional) and column dimension from distinguishing columns
- System validates uniqueness: each (row, col) combination must identify exactly one figure
- On validation success, generates grid

### 6. Figure Grid Display
- CSS grid layout with labeled axes
- Row/column labels truncated to 50 characters
- Empty cells marked with ❌
- Missing files marked with ⚠️
- Figures displayed as embedded images

## Styling

Base font size reduced by 30% (11px body text) via `assets/custom.css` for compact display. Uses Bootstrap theme from `dash-bootstrap-components`.
Filter builder layout is controlled by CSS variables defined near the top of `assets/custom.css` (`--filter-builder-left-gap`, `--filter-badge-offset`); adjust them to change how far the badge sits outside each row.

## Entry Point

CLI launcher at `bin/dashboard.py`:
```bash
python -m feedbax_experiments.bin.dashboard --db-name main --port 8050
```

## Key Design Decisions

**Column name format**: Use `table.column` format throughout to disambiguate joined columns. This is required because both tables have overlapping column names (hash, created_at, etc.).

**Inline builder vs modal**: Filters live in a stacked form where each row exposes dropdowns for Table/Column/Operator/Value (Value uses database-derived choices), and the Figure Type row stays pinned to encourage selecting an identifier without leaving the builder.

**Preset storage**: JSON files rather than database records for simplicity and portability. Each preset file contains a name and filter list.

**Grid validation**: Enforces uniqueness constraint before generating grid to prevent ambiguous figure placement. User must add filters until row/col dimensions uniquely identify records.
