"""Database query engine for filtering and retrieving figure records."""

import json
import logging
from dataclasses import dataclass
from typing import Any, Literal, Optional

import pandas as pd
from sqlalchemy.orm import Session

from feedbax.database import EvaluationRecord, FigureRecord

logger = logging.getLogger(__name__)


@dataclass
class FilterSpec:
    """Specification for a single filter condition."""

    table: Literal["figures", "evaluations"]
    column: str
    operator: Literal["=", "!=", "<", ">", "<=", ">=", "IN", "LIKE", "IS NULL", "IS NOT NULL"]
    value: Any  # Can be None for NULL checks


@dataclass
class ColumnInfo:
    """Information about a column that varies in the filtered recordset."""

    table: str
    column: str
    n_unique: int
    sample_values: list[Any]


@dataclass
class GridData:
    """Data structure for grid rendering."""

    row_values: Optional[list[Any]]  # None if row_col is None
    col_values: list[Any]
    records: dict[tuple[Any, Any], dict[str, Any]]  # (row_val, col_val) -> record dict


class FigureQueryEngine:
    """Handle all database queries and filtering for the dashboard."""

    def __init__(self, session: Session):
        """Initialize the query engine.

        Args:
            session: SQLAlchemy database session
        """
        self.session = session

    def get_filtered_dataframe(
        self,
        filters: list[FilterSpec],
        exclude_archived: bool = True,
    ) -> pd.DataFrame:
        """Execute JOIN query and return filtered results as DataFrame.

        Args:
            filters: List of filter specifications
            exclude_archived: If True, exclude archived figures

        Returns:
            DataFrame with columns prefixed by table name (e.g., "figures.hash")
        """
        # Start with base query joining figures and evaluations
        query = self.session.query(FigureRecord, EvaluationRecord).join(
            EvaluationRecord,
            FigureRecord.evaluation_hash == EvaluationRecord.hash,
        )

        # Exclude archived figures by default
        if exclude_archived:
            query = query.filter(FigureRecord.archived == False)  # noqa: E712

        # Apply filters
        for filter_spec in filters:
            query = self._apply_filter(query, filter_spec)

        # Execute query and convert to DataFrame
        results = query.all()

        if not results:
            # Return empty DataFrame with expected structure
            return pd.DataFrame(columns=["figures.hash", "evaluations.hash"])

        # Convert to rows with prefixed column names
        rows = []
        for fig_record, eval_record in results:
            row = {}

            # Add figure columns with prefix
            for col in fig_record.__table__.columns:
                value = getattr(fig_record, col.name)
                # Serialize JSON columns to string for DataFrame
                if isinstance(value, (list, dict)):
                    value = json.dumps(value)
                row[f"figures.{col.name}"] = value

            # Add evaluation columns with prefix
            for col in eval_record.__table__.columns:
                value = getattr(eval_record, col.name)
                # Serialize JSON columns to string for DataFrame
                if isinstance(value, (list, dict)):
                    value = json.dumps(value)
                row[f"evaluations.{col.name}"] = value

            rows.append(row)

        return pd.DataFrame(rows)

    def _apply_filter(self, query, filter_spec: FilterSpec):
        """Apply a single filter to the query.

        Args:
            query: SQLAlchemy query object
            filter_spec: Filter specification

        Returns:
            Modified query with filter applied
        """
        # Get the appropriate model class
        model_class = FigureRecord if filter_spec.table == "figures" else EvaluationRecord

        # Get the column
        try:
            column = getattr(model_class, filter_spec.column)
        except AttributeError:
            logger.warning(
                f"Column '{filter_spec.column}' not found in table '{filter_spec.table}'"
            )
            return query

        # Handle JSON columns - need to serialize the value for comparison
        if hasattr(column.type, "python_type"):
            # This is a JSON column
            if filter_spec.value is not None and not isinstance(filter_spec.value, str):
                filter_value = json.dumps(filter_spec.value)
            else:
                filter_value = filter_spec.value
        else:
            filter_value = filter_spec.value

        # Apply operator
        if filter_spec.operator == "=":
            query = query.filter(column == filter_value)
        elif filter_spec.operator == "!=":
            query = query.filter(column != filter_value)
        elif filter_spec.operator == "<":
            query = query.filter(column < filter_value)
        elif filter_spec.operator == ">":
            query = query.filter(column > filter_value)
        elif filter_spec.operator == "<=":
            query = query.filter(column <= filter_value)
        elif filter_spec.operator == ">=":
            query = query.filter(column >= filter_value)
        elif filter_spec.operator == "IN":
            if not isinstance(filter_value, (list, tuple)):
                filter_value = [filter_value]
            query = query.filter(column.in_(filter_value))
        elif filter_spec.operator == "LIKE":
            query = query.filter(column.like(f"%{filter_value}%"))
        elif filter_spec.operator == "IS NULL":
            query = query.filter(column.is_(None))
        elif filter_spec.operator == "IS NOT NULL":
            query = query.filter(column.isnot(None))

        return query

    def get_distinguishing_columns(
        self,
        df: pd.DataFrame,
        exclude_columns: Optional[set[str]] = None,
    ) -> list[ColumnInfo]:
        """Get columns that still vary in the filtered DataFrame.

        Args:
            df: Filtered DataFrame
            exclude_columns: Set of columns to exclude from analysis (e.g., grid dimensions)

        Returns:
            List of ColumnInfo objects for columns with >1 unique value
        """
        if df.empty:
            return []

        if exclude_columns is None:
            exclude_columns = set()

        # Columns to always exclude
        auto_exclude = {
            "figures.id",
            "figures.hash",
            "figures.created_at",
            "figures.modified_at",
            "figures.archived",
            "figures.archived_at",
            "evaluations.id",
            "evaluations.hash",
            "evaluations.created_at",
            "evaluations.modified_at",
            "evaluations.archived",
            "evaluations.archived_at",
        }
        exclude_columns = exclude_columns | auto_exclude

        distinguishing = []

        for col in df.columns:
            if col in exclude_columns:
                continue

            # Get unique values (excluding NaN)
            unique_vals = df[col].dropna().unique()

            if len(unique_vals) > 1:
                # Limit sample values to first 5
                sample_values = unique_vals[:5].tolist()

                # Parse table and column name from prefixed column
                table, column = col.split(".", 1)

                distinguishing.append(
                    ColumnInfo(
                        table=table,
                        column=column,
                        n_unique=len(unique_vals),
                        sample_values=sample_values,
                    )
                )

        return distinguishing

    def get_column_values(
        self,
        table: Literal["figures", "evaluations"],
        column: str,
        limit: int = 250,
    ) -> list[Any]:
        """Return distinct values for a column.

        Args:
            table: Table name
            column: Column name
            limit: Max number of unique values to return

        Returns:
            List of distinct non-null values
        """
        model_class = FigureRecord if table == "figures" else EvaluationRecord
        try:
            column_attr = getattr(model_class, column)
        except AttributeError:
            logger.warning(
                "Column '%s' not found in table '%s' when fetching values",
                column,
                table,
            )
            return []

        query = (
            self.session.query(column_attr)
            .filter(column_attr.isnot(None))
            .distinct()
            .limit(limit)
        )

        values = []
        for (value,) in query.all():
            if isinstance(value, (list, dict)):
                values.append(json.dumps(value))
            else:
                values.append(value)
        return values

    def validate_grid_uniqueness(
        self,
        df: pd.DataFrame,
        row_col: Optional[str],
        col_col: str,
    ) -> tuple[bool, str]:
        """Check if row/col combination uniquely identifies records.

        Args:
            df: Filtered DataFrame
            row_col: Row dimension column (prefixed, e.g., "evaluations.train__pert__std")
            col_col: Column dimension column (prefixed)

        Returns:
            (is_valid, error_message)
        """
        if df.empty:
            return False, "No records match the current filters"

        group_cols = [col_col]
        if row_col:
            group_cols.append(row_col)

        # Check for duplicates
        grouped = df.groupby(group_cols).size()

        if (grouped > 1).any():
            # Find which combinations have duplicates
            dupes = grouped[grouped > 1]
            dupe_combos = [f"{combo}: {count} records" for combo, count in dupes.items()]
            return (
                False,
                f"Row/column combination does not uniquely identify records. "
                f"Duplicate combinations: {', '.join(dupe_combos[:3])}...",
            )

        return True, ""

    def build_grid_data(
        self,
        df: pd.DataFrame,
        row_col: Optional[str],
        col_col: str,
    ) -> GridData:
        """Build data structure for grid rendering.

        Args:
            df: Filtered DataFrame
            row_col: Row dimension column (prefixed)
            col_col: Column dimension column (prefixed)

        Returns:
            GridData object with organized records
        """
        if df.empty:
            return GridData(row_values=None, col_values=[], records={})

        # Get unique values for each dimension (sorted)
        col_values = sorted(df[col_col].unique().tolist())
        row_values = sorted(df[row_col].unique().tolist()) if row_col else [None]

        # Build records dictionary
        records = {}
        for row_val in row_values:
            for col_val in col_values:
                # Filter for this specific combination
                if row_col:
                    mask = (df[row_col] == row_val) & (df[col_col] == col_val)
                else:
                    mask = df[col_col] == col_val

                matching = df[mask]

                if len(matching) == 1:
                    # Convert row to dict (single record)
                    record_dict = matching.iloc[0].to_dict()
                    records[(row_val, col_val)] = record_dict

        return GridData(
            row_values=row_values if row_col else None,
            col_values=col_values,
            records=records,
        )

    def get_available_identifiers(self) -> list[str]:
        """Get list of available figure identifiers.

        Returns:
            Sorted list of unique identifiers
        """
        query = self.session.query(FigureRecord.identifier).filter(
            FigureRecord.archived == False  # noqa: E712
        ).distinct()

        identifiers = [row[0] for row in query.all()]
        return sorted(identifiers)

    def get_table_columns(self, table: Literal["figures", "evaluations"]) -> list[str]:
        """Get column names for a given table.

        Args:
            table: Table name

        Returns:
            Sorted list of column names
        """
        model_class = FigureRecord if table == "figures" else EvaluationRecord

        # Exclude internal columns
        exclude_cols = {"id", "created_at", "modified_at", "archived", "archived_at"}

        columns = [
            col.name for col in model_class.__table__.columns
            if col.name not in exclude_cols
        ]

        return sorted(columns)
