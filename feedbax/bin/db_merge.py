#!/usr/bin/env python3
"""
Database merger script that combines multiple SQLite databases.

This script merges multiple SQLite databases using the first database as a template.
It handles schema differences by creating a union of all columns and casting types
to match the template database.

Specific merging rules:
- Models table: Merges all records from all databases
- Evaluations table: Merges only records where expt_name IS NULL
- Figures table: Preserves structure but clears all data
- Other tables: Merges all records (if any)

Usage:
    python scripts/db_merge.py template.db source1.db [source2.db ...] -o output.db
"""

import argparse
import sqlite3
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def get_table_schema(db_path: str, table_name: str) -> List[Dict[str, Any]]:
    """Get schema information for a specific table."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(f"PRAGMA table_info({table_name})")
        columns = []
        for row in cursor:
            columns.append({
                'cid': row[0],
                'name': row[1], 
                'type': row[2],
                'notnull': row[3],
                'dflt_value': row[4],
                'pk': row[5]
            })
        return columns


def get_all_tables(db_path: str) -> Set[str]:
    """Get all table names from a database."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        return {row[0] for row in cursor if not row[0].startswith('sqlite_')}


def analyze_schema_union(db_paths: List[str], table_name: str) -> Dict[str, Any]:
    """Analyze schemas across all databases for a given table and return union info."""
    template_schema = get_table_schema(db_paths[0], table_name)
    template_columns = {col['name']: col for col in template_schema}
    
    all_columns = set(template_columns.keys())
    schema_info = {
        'template_columns': template_columns,
        'all_columns': set(),
        'missing_columns': []
    }
    
    # Collect all unique columns across all databases
    for db_path in db_paths:
        if table_name in get_all_tables(db_path):
            schema = get_table_schema(db_path, table_name)
            for col in schema:
                all_columns.add(col['name'])
                if col['name'] not in template_columns:
                    schema_info['missing_columns'].append(col)
    
    schema_info['all_columns'] = all_columns
    return schema_info


def add_missing_columns(output_db: str, table_name: str, missing_columns: List[Dict]):
    """Add missing columns to the output database."""
    if not missing_columns:
        return
        
    with sqlite3.connect(output_db) as conn:
        for col in missing_columns:
            # Default to JSON type to match template database pattern
            col_type = 'JSON' if col['type'] in ['INTEGER', 'FLOAT', 'VARCHAR'] else col['type']
            sql = f"ALTER TABLE {table_name} ADD COLUMN {col['name']} {col_type}"
            try:
                conn.execute(sql)
                logger.info(f"Added column {col['name']} ({col_type}) to {table_name}")
            except sqlite3.OperationalError as e:
                if "duplicate column name" not in str(e):
                    raise


def generate_cast_expression(column_name: str, source_type: str, target_type: str) -> str:
    """Generate SQL expression for casting between types."""
    # Quote column name if it's a SQL keyword
    quoted_name = f'"{column_name}"' if column_name in ['where', 'order', 'group'] else column_name
    
    if target_type == 'JSON' and source_type in ['INTEGER', 'FLOAT', 'VARCHAR']:
        return f"CASE WHEN {quoted_name} IS NOT NULL THEN json_quote({quoted_name}) ELSE NULL END"
    return quoted_name


def insert_models_records(output_db: str, source_db: str, schema_info: Dict):
    """Insert models records with proper type casting."""
    template_cols = schema_info['template_columns']
    all_cols = schema_info['all_columns']
    
    # Get source schema to determine casting needs
    source_schema = get_table_schema(source_db, 'models')
    source_cols = {col['name']: col for col in source_schema}
    
    # Build column lists and cast expressions
    insert_columns = []
    select_expressions = []
    
    for col_name in sorted(all_cols):
        quoted_col = f'"{col_name}"' if col_name in ['where', 'order', 'group'] else col_name
        
        if col_name == 'id':
            # Generate new IDs
            select_expressions.append("(SELECT COALESCE(MAX(id), 0) FROM models) + ROW_NUMBER() OVER (ORDER BY id) as id")
            insert_columns.append(quoted_col)
        elif col_name in source_cols and col_name in template_cols:
            # Column exists in both, may need casting
            source_type = source_cols[col_name]['type']
            target_type = template_cols[col_name]['type']
            cast_expr = generate_cast_expression(col_name, source_type, target_type)
            select_expressions.append(cast_expr)
            insert_columns.append(quoted_col)
        elif col_name in source_cols:
            # Column only in source, cast to JSON if primitive type
            source_type = source_cols[col_name]['type']
            cast_expr = generate_cast_expression(col_name, source_type, 'JSON')
            select_expressions.append(cast_expr)
            insert_columns.append(quoted_col)
        else:
            # Column only in template or other sources, use NULL
            select_expressions.append('NULL')
            insert_columns.append(quoted_col)
    
    sql = f"""
    ATTACH DATABASE '{source_db}' AS source;
    INSERT INTO models ({', '.join(insert_columns)})
    SELECT {', '.join(select_expressions)}
    FROM source.models
    WHERE hash NOT IN (SELECT hash FROM models);
    DETACH DATABASE source;
    """
    
    with sqlite3.connect(output_db) as conn:
        conn.executescript(sql)
        logger.info(f"Inserted models records from {source_db}")


def insert_evaluations_records(output_db: str, source_db: str, schema_info: Dict):
    """Insert evaluations records where expt_name IS NULL, avoiding duplicates."""
    template_cols = schema_info['template_columns']
    all_cols = schema_info['all_columns']
    
    # Get source schema
    source_schema = get_table_schema(source_db, 'evaluations')
    source_cols = {col['name']: col for col in source_schema}
    
    # Build column lists and cast expressions
    insert_columns = []
    select_expressions = []
    
    for col_name in sorted(all_cols):
        quoted_col = f'"{col_name}"' if col_name in ['where', 'order', 'group'] else col_name
        
        if col_name == 'id':
            select_expressions.append("(SELECT COALESCE(MAX(id), 0) FROM evaluations) + ROW_NUMBER() OVER (ORDER BY id) as id")
            insert_columns.append(quoted_col)
        elif col_name in source_cols and col_name in template_cols:
            source_type = source_cols[col_name]['type']
            target_type = template_cols[col_name]['type']
            cast_expr = generate_cast_expression(col_name, source_type, target_type)
            select_expressions.append(cast_expr)
            insert_columns.append(quoted_col)
        elif col_name in source_cols:
            source_type = source_cols[col_name]['type']
            cast_expr = generate_cast_expression(col_name, source_type, 'JSON')
            select_expressions.append(cast_expr)
            insert_columns.append(quoted_col)
        else:
            select_expressions.append('NULL')  
            insert_columns.append(quoted_col)
    
    sql = f"""
    ATTACH DATABASE '{source_db}' AS source;
    INSERT INTO evaluations ({', '.join(insert_columns)})
    SELECT {', '.join(select_expressions)}
    FROM source.evaluations 
    WHERE expt_name IS NULL 
    AND hash NOT IN (SELECT hash FROM evaluations);
    DETACH DATABASE source;
    """
    
    with sqlite3.connect(output_db) as conn:
        conn.executescript(sql)
        logger.info(f"Inserted evaluations records (expt_name IS NULL) from {source_db}")


def clear_figures_table(output_db: str):
    """Clear all records from figures table while preserving structure."""
    with sqlite3.connect(output_db) as conn:
        conn.execute("DELETE FROM figures")
        logger.info("Cleared figures table data")


def merge_databases(template_db: str, source_dbs: List[str], output_db: str):
    """Main database merging logic."""
    # Copy template database to output
    logger.info(f"Copying template database {template_db} to {output_db}")
    shutil.copy2(template_db, output_db)
    
    # Get all tables from all databases
    all_tables = set()
    for db_path in [template_db] + source_dbs:
        all_tables.update(get_all_tables(db_path))
    
    logger.info(f"Found tables: {sorted(all_tables)}")
    
    # Process each table
    for table_name in sorted(all_tables):
        logger.info(f"Processing table: {table_name}")
        
        # Analyze schema union
        db_paths = [template_db] + source_dbs
        schema_info = analyze_schema_union(db_paths, table_name)
        
        # Add missing columns to output database
        add_missing_columns(output_db, table_name, schema_info['missing_columns'])
        
        # Insert records from each source database
        for source_db in source_dbs:
            if table_name not in get_all_tables(source_db):
                continue
                
            if table_name == 'models':
                insert_models_records(output_db, source_db, schema_info)
            elif table_name == 'evaluations':
                insert_evaluations_records(output_db, source_db, schema_info)
            # For other tables, could add default merge logic here
    
    # Clear figures table as requested
    if 'figures' in all_tables:
        clear_figures_table(output_db)
    
    # Verify integrity
    with sqlite3.connect(output_db) as conn:
        result = conn.execute("PRAGMA integrity_check").fetchone()[0]
        if result == 'ok':
            logger.info("Database integrity check passed")
        else:
            logger.error(f"Database integrity check failed: {result}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple SQLite databases using the first as a template."
    )
    parser.add_argument('databases', nargs='+', 
                       help='Database files to merge (first is template)')
    parser.add_argument('-o', '--output', required=True,
                       help='Output database file path')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite output file if it exists')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate inputs
    if len(args.databases) < 2:
        logger.error("At least 2 databases required (template + 1 source)")
        sys.exit(1)
    
    for db_path in args.databases:
        if not Path(db_path).exists():
            logger.error(f"Database file not found: {db_path}")
            sys.exit(1)
    
    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        logger.error(f"Output file exists: {args.output}. Use --overwrite to replace.")
        sys.exit(1)
    
    # Perform merge
    template_db = args.databases[0]
    source_dbs = args.databases[1:]
    
    logger.info(f"Merging {len(source_dbs)} source databases into template {template_db}")
    logger.info(f"Output: {args.output}")
    
    try:
        merge_databases(template_db, source_dbs, args.output)
        
        # Report final counts
        with sqlite3.connect(args.output) as conn:
            for table in ['models', 'evaluations', 'figures']:
                try:
                    count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                    logger.info(f"Final {table} count: {count}")
                except sqlite3.OperationalError:
                    pass
        
        logger.info("Database merge completed successfully")
        
    except Exception as e:
        logger.error(f"Merge failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()