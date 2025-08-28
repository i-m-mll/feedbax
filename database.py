"""
Database tools for cataloguing trained models and notebook evaluations/figures.

Written with the help of Claude 3.5 Sonnet.
"""

import hashlib
import json
import logging
import uuid
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Literal, Optional, TypeVar

import equinox as eqx
import jax.random as jr
import jax.tree as jt
import jax_cookbook.tree as jtree
import matplotlib.figure as mplf
import plotly
import plotly.graph_objects as go
import pyexiv2
from alembic.migration import MigrationContext
from alembic.operations import Operations
from feedbax.misc import attr_str_tree_to_where_func
from jax_cookbook import allf, arrays_to_lists, is_not_type, is_type, save
from jaxtyping import PyTree
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    ColumnElement,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Table,
    create_engine,
    inspect,
    or_,
    update,
)
from sqlalchemy.dialects.postgresql import array as dbarray
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)
from sqlalchemy.sql import func
from sqlalchemy.sql.type_api import TypeEngine

from feedbax_experiments.config import PATHS, STRINGS
from feedbax_experiments.config.yaml import get_yaml_loader
from feedbax_experiments.hyperparams import (
    cast_hps,
    flatten_hps,
    load_hps,
    take_train_histories_hps,
)
from feedbax_experiments.misc import (
    exclude_unshared_keys_and_identical_values,
    get_md5_hexdigest,
    with_caller_logger,
)
from feedbax_experiments.tree_utils import pp
from feedbax_experiments.types import (
    LDict,
    TreeNamespace,
    dict_to_namespace,
    is_dict_with_int_keys,
    namespace_to_dict,
    unflatten_dict_keys,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# Prevent alembic from polluting the console with routine migration logs
logging.getLogger("alembic.runtime.migration").setLevel(logging.WARNING)


class RecordBase(DeclarativeBase):
    type_annotation_map = {
        dict[str, Any]: JSON,
        dict[str, str]: JSON,
        Sequence[str]: JSON,
        Sequence[int]: JSON,
        Sequence[float]: JSON,
        dict[str, Sequence[str]]: JSON,
    }


BaseT = TypeVar("BaseT", bound=RecordBase)


class ModelRecord(RecordBase):
    __tablename__ = STRINGS.db_table_names.models

    id: Mapped[int] = mapped_column(primary_key=True)
    hash: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    is_path_defunct: Mapped[bool] = mapped_column(default=False)
    version_info: Mapped[Optional[dict[str, str]]]

    postprocessed: Mapped[bool] = mapped_column(default=False)
    has_replicate_info: Mapped[bool]
    expt_name: Mapped[str]

    # Explicitly define some parameter columns to avoid typing issues, though our dynamic column
    # migration would handle whatever parameters the user happens to pass, without this.
    model__n_replicates: Mapped[int]
    pert__type: Mapped[str]
    pert__std: Mapped[float]
    where: Mapped[dict[str, Sequence[str]]]
    n_batches: Mapped[int]
    save_model_parameters: Mapped[Sequence[int]]

    @hybrid_property
    def path(self):
        return get_hash_path(PATHS.models, self.hash)

    @hybrid_property
    def replicate_info_path(self):
        if self.has_replicate_info:
            return get_hash_path(
                PATHS.models, self.hash, suffix=STRINGS.file_suffixes.replicate_info
            )
        else:
            return None

    @hybrid_property
    def train_history_path(self):
        return get_hash_path(PATHS.models, self.hash, suffix=STRINGS.file_suffixes.train_history)

    @hybrid_property
    def where_train(self):
        return {int(i): attr_str_tree_to_where_func(strs) for i, strs in self.where.items()}


MODEL_RECORD_BASE_ATTRS = [
    "id",
    "hash",
    "created_at",
    "expt_name",
    "is_path_defunct",
    "has_replicate_info",
]


class EvaluationRecord(RecordBase):
    """Represents a single evaluation."""

    __tablename__ = STRINGS.db_table_names.evaluations

    # model = relationship("ModelRecord")
    figures = relationship("FigureRecord", back_populates="evaluation")

    id: Mapped[int] = mapped_column(primary_key=True)
    hash: Mapped[str] = mapped_column(unique=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    modified_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    expt_name: Mapped[Optional[str]]
    # model_hash: Mapped[Optional[str]] = mapped_column(ForeignKey(f'{MODELS_TABLE_NAME}.hash'))
    model_hashes: Mapped[Optional[Sequence[str]]] = mapped_column(nullable=True)
    archived: Mapped[bool] = mapped_column(default=False)
    archived_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    version_info_eval: Mapped[Optional[dict[str, str]]]

    @hybrid_property
    def figure_dir(self):
        return PATHS.figures / self.hash


class FigureRecord(RecordBase):
    """Represents a figure generated during evaluation."""

    __tablename__ = STRINGS.db_table_names.figures

    evaluation = relationship("EvaluationRecord", back_populates="figures")
    # model = relationship("ModelRecord")

    id: Mapped[int] = mapped_column(primary_key=True)
    hash: Mapped[str] = mapped_column(unique=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    modified_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    evaluation_hash: Mapped[str] = mapped_column(
        ForeignKey(f"{STRINGS.db_table_names.evaluations}.hash")
    )
    identifier: Mapped[str]
    figure_type: Mapped[str]
    saved_formats: Mapped[Sequence[str]]
    archived: Mapped[bool] = mapped_column(default=False)
    archived_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)

    model_hashes: Mapped[Optional[Sequence[str]]] = mapped_column(nullable=True)

    # These are also redundant, and can be inferred from `evaluation_hash`
    pert__type: Mapped[str] = mapped_column(nullable=True)
    pert__std: Mapped[float] = mapped_column(nullable=True)
    # pert__stds: Mapped[Sequence[float]] = mapped_column(nullable=True)


TABLE_NAME_TO_MODEL = {
    mapper.class_.__tablename__: mapper.class_ for mapper in RecordBase.registry.mappers
}


def get_sql_type(value) -> TypeEngine:
    if isinstance(value, bool):
        return Boolean()
    elif isinstance(value, int):
        return Integer()
    elif isinstance(value, float):
        return Float()
    elif isinstance(value, str):
        return String()
    else:
        return JSON()


def update_table_schema(engine, table_name: str, columns: Dict[str, Any], all_json: bool = False):
    """Dynamically add new columns using Alembic operations."""
    RecordBase.metadata.create_all(engine)

    # Get existing columns
    inspector = inspect(engine)
    existing_columns = {col["name"] for col in inspector.get_columns(table_name)}

    # Create Alembic context
    context = MigrationContext.configure(engine.connect())
    op = Operations(context)
    model_class = TABLE_NAME_TO_MODEL[table_name]

    # Add only new columns using Alembic operations
    for key, value in columns.items():
        if key not in existing_columns:
            column_type = JSON() if all_json else get_sql_type(value)
            column = Column(key, column_type, nullable=True)
            setattr(model_class, key, column)
            op.add_column(table_name, column)

    RecordBase.metadata.clear()  # Clear SQLAlchemy's cached schema
    RecordBase.metadata.create_all(engine)  # Recreate tables with new schema


def init_db_session(db_path: str = "sqlite:///models.db"):
    """Opens a session to an SQLite database.

    If the database/file does not exist, it will be created.

    !!! dev
        If any of the tables in the database contain columns which are not found in the
        definition of their respective SQLAlchemy models (e.g. `Model` for the
        models table), dynamically add those column(s) to the model class(es) upon
        starting.

        During development I paired this with `update_table_schema` to automatically add
        columns to the tables, when previously unseen hyperparameters were passed by the user.

        At first I was mildly concerned that this would lead to some corruption or other
        problems due to (say) accidentally breaking the schema, but I have had no problems
        yet.

        TODO: Still, the list of known columns is at this point pretty static for this project,
        so I could explicitly add all of them to the model classes.
    """
    engine = create_engine(db_path)
    RecordBase.metadata.create_all(engine)

    # Dynamically add missing columns to the table record classes
    inspector = inspect(engine)
    for table_name in inspector.get_table_names():
        existing_columns = inspector.get_columns(table_name)

        model = TABLE_NAME_TO_MODEL[table_name]

        for col in existing_columns:
            try:
                getattr(model, col["name"])
            except AttributeError:
                setattr(
                    model,
                    col["name"],
                    Column(col["name"], col["type"], nullable=col["nullable"]),
                )

    RecordBase.metadata.clear()  # Clear SQLAlchemy's cached schema
    RecordBase.metadata.create_all(engine)  # Recreate tables with new schema

    return sessionmaker(bind=engine)()


def get_db_session(name: str = "main"):
    """Create a database session for the project database with the given name."""
    return init_db_session(f"sqlite:///{PATHS.db}/{name}.db")


@contextmanager
def db_session(name: str = "main", autocommit: bool = True) -> Iterator[Session]:
    """
    Usage:
        with db_session("main") as db:
            db.add(obj)
            # no commit needed if autocommit=True

        with db_session("main", autocommit=False) as db:
            db.add(obj)
            db.commit()  # you control commits
    """
    db = get_db_session(name)  # your existing factory
    check_model_files(db)
    try:
        if autocommit:
            # opens a transaction; commits on success, rolls back on error
            with db.begin():
                yield db
        else:
            # manual control: you call db.commit() / db.rollback()
            yield db
    except Exception:
        if not autocommit:
            db.rollback()
        raise
    finally:
        db.close()


def hash_file(path: Path) -> str:
    """Generate MD5 hash of file."""
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)
    return md5.hexdigest()


def generate_temp_path(directory: Path, prefix: str = "temp_", suffix: str = ".eqx") -> Path:
    """Generate a temporary file path."""
    return directory / f"{prefix}{uuid.uuid4()}{suffix}"


def query_model_records(
    session: Session,
    filters: Optional[Dict[str, Any]] = None,
    match_all: bool = True,
    exclude_defunct: bool = True,
) -> list[ModelRecord]:
    """Query model records from database matching filter criteria."""
    if filters is None:
        filters = {}

    if exclude_defunct:
        # Don't check if defunct here, otherwise we'll be checking the
        # DB and littering the logs with info from `check_model_files`
        filters["is_path_defunct"] = False

    return query_records(session, ModelRecord, filters, match_all)


class AlwaysEquatesFalse:
    """Objects of this class always compare `False` for equality, except against themselves."""

    def __eq__(self, other):
        return isinstance(other, AlwaysEquatesFalse)

    def __req__(self, other):
        return isinstance(other, AlwaysEquatesFalse)


def _make_filter_condition(
    model_class: type["BaseT"], key: str, value: Any
) -> ColumnElement | bool:
    """
    Return a SQLAlchemy filter expression for `model_class.key == value`,
    JSON-dumping `value` if needed, or `False` if the column doesn’t exist.
    """
    column = getattr(model_class, key, None)
    if column is None:
        # no such column → always-false
        return False

    if isinstance(column.type, JSON):
        value = json.dumps(value)

    return column == value


def query_records(
    session: Session,
    record_type: str | type[BaseT],
    filters: Optional[Dict[str, Any]] = None,
    match_all: bool = True,
) -> list[BaseT]:
    """Query records from database matching filter criteria.

    Args:
        session: Database session
        record_type: SQLAlchemy model class or table name to query
        filters: Dictionary of {column: value} pairs to filter by
        match_all: If True, return only records matching all filters (AND).
                  If False, return records matching any filter (OR).
    """
    model_class: type[BaseT] = get_model_class(record_type)
    query = session.query(model_class)

    if filters:
        conditions = [
            # If the column is not found, that counts as `False`
            _make_filter_condition(model_class, key, value)
            for key, value in filters.items()
        ]

        # ? Can we skip the query if any of the conditions are already `False`?
        if match_all:
            for condition in conditions:
                query = query.filter(condition)
        else:
            query = query.filter(or_(*conditions))

    return query.all()


def _valid_filter_key(k: str) -> bool:
    """Check if a key is a valid filter key for a record."""
    return not k.startswith("_") and k not in ["id", "hash", "created_at"]


def get_record(
    session: Session,
    record_type: str | type[BaseT],
    enforce_unique: bool = True,
    **filters: Any,
) -> Optional[BaseT]:
    """Get single record matching all filters exactly.

    Args:
        session: Database session
        record_type: SQLAlchemy model class or table name to query
        **filters: Column=value pairs to filter by

    Raises:
        ValueError: If multiple matches found or unknown table name
    """
    model_class: type[BaseT] = get_model_class(record_type)
    matches = query_records(session, model_class, filters)
    if not matches:
        return None
    if enforce_unique and len(matches) > 1:
        ids_str = ", ".join(str(match.id) for match in matches)  # type: ignore
        all_unfiltered_params = [
            {k: v for k, v in match.__dict__.items() if _valid_filter_key(k) and k not in filters}
            for match in matches
        ]
        all_disparate_params = exclude_unshared_keys_and_identical_values(all_unfiltered_params)
        err_msg = (
            f"Multiple {model_class.__name__}s (record id: {ids_str}) found matching filters. "
            + "The following key-value pairs distinguish each match:\n\t"
            + "\n\t".join([f"{i}. {params}" for i, params in enumerate(all_disparate_params)])
        )
        raise ValueError(err_msg)
    return matches[0]


def get_model_record(
    session: Session,
    exclude_defunct: bool = True,
    **filters: Any,
) -> Optional[ModelRecord]:
    """Get single model record matching all filters exactly.

    Args:
        session: Database session
        exclude_defunct: If True, only consider models with accessible files
        **filters: Column=value pairs to filter by

    Returns:
        Matching record, or None if not found

    Raises:
        ValueError: If multiple matches found
    """
    if exclude_defunct:
        filters["is_path_defunct"] = False

    return get_record(session, ModelRecord, **filters)


def get_model_class(record_type: str | type[BaseT]) -> type[BaseT]:
    """Convert table name to model class if needed."""
    if isinstance(record_type, str):
        if record_type not in TABLE_NAME_TO_MODEL:
            raise ValueError(f"Unknown table name: {record_type}")
        return TABLE_NAME_TO_MODEL[record_type]
    return record_type


def get_hash_path(
    directory: Path, hash_: str, suffix: Optional[str] = None, ext: str = ".eqx"
) -> Path:
    if suffix is None:
        suffix = ""
    components = [hash_, suffix]
    return (directory / "_".join(c for c in components if c)).with_suffix(ext)


def check_model_files(
    session: Session,
    clean_orphaned_files: Literal["no", "delete", "archive"] = "no",
) -> None:
    """Check model files and update availability status."""
    logger.debug("Checking availability of model files...")

    try:
        records = session.query(ModelRecord).all()
        known_hashes = {record.hash for record in records}

        for record in records:
            model_file_exists = get_hash_path(PATHS.models, record.hash).exists()
            replicate_info_file_exists = get_hash_path(
                PATHS.models,
                record.hash,
                suffix=STRINGS.file_suffixes.replicate_info,
            ).exists()

            if record.is_path_defunct and model_file_exists:
                logger.info(f"File found for defunct model record {record.hash}; restored")
            elif not record.is_path_defunct and not model_file_exists:
                logger.info(f"File missing for model {record.hash}; marked as defunct")

            record.is_path_defunct = not model_file_exists
            record.has_replicate_info = replicate_info_file_exists

        if clean_orphaned_files != "no":
            archive_dir = PATHS.models.parent / f"{PATHS.models.name}_archive"
            archive_dir.mkdir(exist_ok=True)
            for file_path in PATHS.models.glob("*.eqx"):
                # Take hash as the first part of the filename (i.e. ignore "replicate_info" etc.)
                file_hash = file_path.stem.split("_")[0]
                if file_hash not in known_hashes:
                    if clean_orphaned_files == "delete":
                        logger.info(f"Deleting orphaned file: {file_path}")
                        file_path.unlink()
                    elif clean_orphaned_files == "archive":
                        logger.info(f"Moving orphaned file to archive: {file_path}")
                        file_path.rename(archive_dir / file_path.name)

        session.commit()
        logger.debug("Finished checking model files")

    except Exception as e:
        session.rollback()
        logger.error(f"Error checking model files: {e}")
        raise e


def replace_in_column(
    session: Session,
    table_or_model: type,  # Can be a Table object or a Declarative ORM model class
    column_name: str,
    find_value: str,
    replace_value: str,
) -> None:
    """
    Replaces all occurrences of `find_value` with `replace_value` in the specified
    `column_name` of `table_or_model`.

    Arguments:
        session: The SQLAlchemy session to use for the database operation.
        table_or_model: The SQLAlchemy Table object or ORM model class.
        column_name: The name of the string column to update.
        find_value: The string value to find and replace.
        replace_value: The string value to replace with.
    """
    if isinstance(table_or_model, Table):
        table_to_update = table_or_model
        column_key = getattr(table_or_model.c, column_name)
    elif (
        isinstance(table_or_model, type)  # Check if it's a class
        and hasattr(table_or_model, "__table__")
        and isinstance(getattr(table_or_model, "__table__"), Table)
    ):
        table_to_update = getattr(table_or_model, "__table__")
        column_key = getattr(table_or_model, column_name)
    else:
        raise TypeError(
            "table_or_model must be a SQLAlchemy Table object or a mapped ORM model class. "
            f"Received type: {type(table_or_model)}"
        )

    stmt = update(table_to_update).values(
        {column_key: func.replace(column_key, find_value, replace_value)}
    )
    session.execute(stmt)


def yaml_dump(f, data: Any):
    """Custom YAML dump that ends in a group separator character.

    The group separator indicates where we should stop reading YAML from the
    `.eqx` file, and move on to model deserialisation.

    Importantly, the string output of `yaml.dump` does not need to be on a single line,
    for this to work.
    """
    yaml = get_yaml_loader(typ="safe")
    yaml.dump(data, f)
    f.write(f"\n{chr(STRINGS.serialisation.sep_chr)}\n".encode())


def save_tree(
    tree: PyTree,
    directory: Path,
    hps: TreeNamespace = TreeNamespace(),
    hash_: Optional[str] = None,
    suffix: Optional[str] = None,
    **kwargs,
) -> tuple[str, Path]:
    """Save object to file whose name is its hash.

    If `hash_` is passed, save to the file with the corresponding
    filename, ending in `suffix`,
    """

    if hash_ is not None:
        if suffix is None:
            raise ValueError("If `hash_` is provided, `suffix` must also be provided.")
        path = get_hash_path(directory, hash_, suffix=suffix, **kwargs)
    else:
        path = generate_temp_path(directory)

    save(
        path,
        tree,
        hyperparameters=namespace_to_dict(hps),
        dump_fn=yaml_dump,
    )

    if hash_ is not None:
        file_hash = hash_
        final_path = path
    else:
        # ? Alternatively, could compute hashes on (a subset of) hps, especially if we don't want the hash
        # ? to depend on certain things (e.g. version info)
        file_hash = hash_file(path)
        final_path = get_hash_path(directory, file_hash, **kwargs)
        path.rename(final_path)

    return file_hash, final_path


def _read_until_special(file, special_char):
    result = []
    for line in file:
        if line.strip().decode("ascii") == special_char:
            return "".join(result)
        result.append(line.decode("ascii"))
    raise ValueError("Malformed serialisation!")


def load_tree_with_hps(
    path: Path,
    setup_tree_func: Callable,
    **kwargs,
) -> tuple[PyTree, TreeNamespace]:
    """Similar to `feedbax.load_with_hyperparameters, but for namespace-based hyperparameters"""
    yaml = get_yaml_loader(typ="safe")
    with open(path, "rb") as f:
        hps_dict = yaml.load(_read_until_special(f, chr(STRINGS.serialisation.sep_chr)))

        hps = dict_to_namespace(hps_dict, to_type=TreeNamespace, exclude=is_dict_with_int_keys)

        # Initialization key isn't important
        tree = setup_tree_func(hps, key=jr.PRNGKey(0))
        tree = eqx.tree_deserialise_leaves(f, tree, **kwargs)

    return tree, hps


def load_tree_without_hps(
    path: Path,
    hps: TreeNamespace,
    setup_tree_func: Callable,
    **kwargs,
) -> PyTree:
    """Load tree using provided hyperparameters, skipping file's hyperparameter section.

    This function is useful when you want to use hyperparameters from the database
    instead of the serialized hyperparameters in the file, ensuring compatibility
    when new training hyperparameters are added after model training.

    Args:
        path: Path to the serialized model file
        hps: Hyperparameters to use for tree construction
        setup_tree_func: Function to create the tree structure
        **kwargs: Additional arguments for eqx.tree_deserialise_leaves

    Returns:
        The deserialized tree using the provided hyperparameters
    """
    with open(path, "rb") as f:
        # Skip the hyperparameter section by reading until separator
        _read_until_special(f, chr(STRINGS.serialisation.sep_chr))

        # Setup tree using provided hps
        tree = setup_tree_func(hps, key=jr.PRNGKey(0))
        tree = eqx.tree_deserialise_leaves(f, tree, **kwargs)

    return tree


def save_model_and_add_record(
    session: Session,
    model: Any,
    hps_train: TreeNamespace,
    train_history: Optional[Any] = None,
    replicate_info: Optional[Any] = None,
    version_info: Optional[Dict[str, str]] = None,
) -> ModelRecord:
    """Save model files with hash-based names and add database record."""

    hps_train = arrays_to_lists(hps_train)

    # Replace LDict with plain dict so it is serialisable
    record_hps = flatten_hps(hps_train, ldict_to_dict=True) | dict(version_info=version_info)
    # If any LDicts have made their way here,
    record_params = namespace_to_dict(record_hps)

    # Save model and get hash-based filename
    # TODO: Optionally, let the hash/existence checks be independent of `version_info`
    # i.e. so we don't retrain models just because Equinox got a minor update or something
    # (alternatively, could just fix the package versions for the project)
    model_hash, _ = save_tree(model, PATHS.models, hps_train)

    # Save associated files if provided
    for tree, suffix in (
        (train_history, STRINGS.file_suffixes.train_history),
        (replicate_info, STRINGS.file_suffixes.replicate_info),
    ):
        if tree is not None:
            save_tree(tree, PATHS.models, hps_train, hash_=model_hash, suffix=suffix)

    update_table_schema(
        session.bind,
        STRINGS.db_table_names.models,
        record_params,
        all_json=True,
    )

    # Create database record
    model_record = ModelRecord(
        hash=model_hash,
        is_path_defunct=False,
        has_replicate_info=replicate_info is not None,
        **record_params,
    )

    # Delete existing record with same hash, if it exists
    existing_record = get_record(session, ModelRecord, hash=model_hash)
    if existing_record is not None:
        session.delete(existing_record)
        session.commit()
        logger.debug(f"Replacing existing model record with hash {model_hash}")

    session.add(model_record)
    session.commit()
    return model_record


def generate_eval_hash(
    model_hashes: Optional[Sequence[str]],
    eval_params: Dict[str, Any],
    expt_name: Optional[str] = None,
) -> str:
    """Generate a hash for a notebook evaluation based on model hash and parameters.

    Args:
        model_hash: Hash of the model being evaluated. None for training notebooks.
        eval_params: Parameters used for evaluation
    """
    if model_hashes is None:
        model_str = "None"
    else:
        model_str = "".join(model_hashes)

    eval_str = "_".join(
        [
            model_str,
            f"{expt_name or 'None'}",
            f"{json.dumps(eval_params, sort_keys=True)}",
        ]
    )
    return get_md5_hexdigest(eval_str)


def add_evaluation(
    session: Session,
    models: PyTree[ModelRecord],
    eval_parameters: Dict[str, Any],
    expt_name: Optional[str] = None,
    version_info: Optional[dict[str, str]] = None,
) -> EvaluationRecord:
    """Create new notebook evaluation record.

    Args:
        session: Database session
        model_id: ID of the model used (None for training notebooks)
        expt_name: Name identifying the analysis/experiment
        eval_parameters: Parameters used for evaluation
    """
    eval_parameters = arrays_to_lists(eval_parameters)

    if models is None:
        model_hashes = None
    else:
        model_hashes = [model.hash for model in jt.leaves(models, is_leaf=is_type(ModelRecord))]

    # Generate hash from model_id (if any) and parameters
    eval_hash = generate_eval_hash(
        model_hashes=model_hashes,
        eval_params=eval_parameters,
        expt_name=expt_name,
    )

    # Migrate the evaluations table so it has all the necessary columns
    update_table_schema(
        session.bind,
        STRINGS.db_table_names.evaluations,
        eval_parameters,
        all_json=True,
    )

    figure_dir = PATHS.figures / eval_hash
    figure_dir.mkdir(exist_ok=True)

    # quarto_output_dir = QUARTO_OUT_DIR / eval_hash
    # quarto_output_dir.mkdir(exist_ok=True)

    # Delete existing record with same hash, if it exists
    existing_record = get_record(session, EvaluationRecord, hash=eval_hash)
    if existing_record is not None:
        existing_record.modified_at = datetime.utcnow()
        logger.info(f"Updating timestamp of existing evaluation record with hash {eval_hash}")
        eval_record = existing_record
    else:
        eval_record = EvaluationRecord(
            hash=eval_hash,
            model_hashes=model_hashes,  # Can be None
            version_info_eval=version_info,
            **eval_parameters,
        )
        session.add(eval_record)

    session.commit()
    return eval_record


def generate_figure_hash(eval_hash: str, identifier: str, parameters: Dict[str, Any]) -> str:
    """Generate hash for a figure based on evaluation, identifier, and parameters."""
    figure_str = f"{eval_hash}_{identifier}_{json.dumps(parameters, sort_keys=True)}"
    return get_md5_hexdigest(figure_str)


EXTS_WITH_EXIF = ["jpg", "jpeg", "tif", "tiff", "webp"]


pyexiv2.registerNs("http://example.com/ns/custom/", "custom")


@with_caller_logger
def savefig(
    fig,
    label,
    fig_dir: Path,
    image_formats: Sequence[str],
    transparent=True,
    metadata: Optional[dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
    **kwargs,
):
    path = str(fig_dir / f"{label}") + ".{ext}"

    if isinstance(fig, mplf.Figure):
        for ext in image_formats:
            fig.savefig(
                path.format(ext=ext),
                transparent=transparent,
                **kwargs,
            )

    elif isinstance(fig, go.Figure):
        fig.update_layout(meta=metadata)

        for ext in image_formats:
            path_i = path.format(ext=ext)
            if ext == "html":
                fig.write_html(path_i, **kwargs)
            elif ext == "json":
                fig.write_json(path_i, engine="auto", **kwargs)
            else:
                width = getattr(fig.layout, "width", None)
                height = getattr(fig.layout, "height", None)
                fig.write_image(path_i, scale=2, width=width, height=height, **kwargs)

                if metadata is not None and ext in EXTS_WITH_EXIF:
                    try:
                        img = pyexiv2.Image(path_i)
                        metadata_xmp = {f"Xmp.custom.{k}": v for k, v in metadata.items()}
                        metadata_xmp["Xmp.dc.description"] = json.dumps(metadata, indent=2)
                        img.modify_xmp(metadata_xmp)
                        img.close()
                    except Exception as e:
                        raise (e)
                        logger.error(f"Failed to save metadata for image at {path_i}: {e}")


def add_evaluation_figure(
    session: Session,
    eval_record: EvaluationRecord,
    figure: go.Figure | mplf.Figure,
    identifier: str,
    model_records: PyTree[ModelRecord] = None,
    save_formats: Optional[str | Sequence[str]] = "png",
    skip_schema_update: bool = False,
    **parameters: Any,
) -> FigureRecord:
    """Save figure and create database record with dynamic parameters.

    Args:
        session: Database session
        eval_record: Evaluation record with which the figure is associated
        figure: Plotly or matplotlib figure to save
        identifier: Unique label for this type of figure
        save_formats: The image types to save.
        **parameters: Additional parameters that distinguish the figure
    """
    parameters = arrays_to_lists(parameters)

    # Generate hash including parameters
    figure_hash = generate_figure_hash(eval_record.hash, identifier, parameters)

    if isinstance(save_formats, str):
        save_formats_set = {save_formats}
    elif isinstance(save_formats, Sequence):
        save_formats_set = set(save_formats)
    elif save_formats is None:
        save_formats_set: set[str] = set()

    # Always save JSON for database recorded figures
    save_formats_set.add("json")

    # Maybe the user passed an extension with a leading dot
    save_formats_set = {format.strip(".") for format in save_formats_set}

    if isinstance(figure, mplf.Figure):
        figure_type = "matplotlib"
    elif isinstance(figure, go.Figure):
        figure_type = "plotly"

    # Save figure in subdirectory with same hash as evaluation
    eval_record.figure_dir.mkdir(exist_ok=True)

    savefig(figure, figure_hash, eval_record.figure_dir, list(save_formats_set))

    # Update schema with new parameters
    if not skip_schema_update:
        update_table_schema(
            session.bind,
            STRINGS.db_table_names.figures,
            parameters,
            all_json=True,
        )

    if model_records is None:
        model_hashes = None
    else:
        model_hashes = [
            model.hash for model in jt.leaves(model_records, is_leaf=is_type(ModelRecord))
        ]

    #! TODO: Implement `modified_at` rather than just replace it entirely.
    figure_record = FigureRecord(
        hash=figure_hash,
        evaluation_hash=eval_record.hash,
        model_hashes=model_hashes,
        identifier=identifier,
        figure_type=figure_type,
        saved_formats=save_formats,
        **parameters,
    )

    # Replace existing record if it exists
    existing_record = get_record(session, FigureRecord, hash=figure_hash)
    if existing_record is not None:
        session.delete(existing_record)
        session.commit()
        logger.debug(f"Replacing existing figure record with hash {figure_hash}")

    session.add(figure_record)
    session.commit()
    return figure_record


def use_record_params_where_none(parameters: dict[str, Any], record: RecordBase) -> dict[str, Any]:
    """Helper to replace `None` values in `parameters` with matching values from `record`.

    Will raise an error if `parameters` contains any keys that are not columns in the type of `record`.
    """
    return {k: getattr(record, k) if v is None else v for k, v in parameters.items()}


def archive_orphaned_records(session: Session) -> None:
    """Mark records as archived if their model references no longer exist."""
    logger.info("Checking for orphaned records...")

    try:
        # Get all existing model hashes
        model_hashes = [r.hash for r in session.query(ModelRecord).all()]  # Changed to a list

        # Find and archive orphaned evaluation records
        orphaned_evals = (
            session.query(EvaluationRecord)
            .filter(
                EvaluationRecord.model_hashes.isnot(None),  # Skip training evals
                EvaluationRecord.archived == False,
                ~EvaluationRecord.model_hashes.op("<@")(
                    dbarray(model_hashes)
                ),  # Check if any model hash is not in model_hashes
            )
            .all()
        )

        if orphaned_evals:
            now = datetime.utcnow()
            for record in orphaned_evals:
                assert record.model_hashes is not None, (
                    "Evaluation records without model hashes should be filtered out already!"
                )
                missing_hashes = set(record.model_hashes) - set(model_hashes)
                logger.warning(
                    f"Archiving evaluation {record.hash} - referenced model(s) "
                    f"{', '.join(missing_hashes)} no longer exist"
                )
                record.archived = True
                record.archived_at = now

                # Also archive associated figures
                for figure in record.figures:
                    if not figure.archived:
                        figure.archived = True
                        figure.archived_at = now

            session.commit()

        # Find and archive orphaned figure records
        orphaned_figures = (
            session.query(FigureRecord)
            .filter(
                FigureRecord.model_hashes.isnot(None),
                FigureRecord.archived == False,  # noqa: E712
                ~FigureRecord.model_hashes.op("<@")(
                    dbarray(model_hashes)
                ),  # Similar check for figures
            )
            .all()
        )

        if orphaned_figures:
            now = datetime.utcnow()
            for record in orphaned_figures:
                assert record.model_hashes is not None, (
                    "Figure records without model hashes should be filtered out already!"
                )
                missing_hashes = set(record.model_hashes) - set(model_hashes)
                logger.warning(
                    f"Archiving figure {record.hash} - referenced model(s) "
                    f"{', '.join(missing_hashes)} no longer exist"
                )
                record.archived = True
                record.archived_at = now
            session.commit()

        if not (orphaned_evals or orphaned_figures):
            logger.info("No orphaned records found")

    except Exception as e:
        session.rollback()
        logger.error(f"Error archiving orphaned records: {e}")
        raise


RecordDict = jtree.make_named_dict_subclass("RecordDict")
ColumnTuple = jtree.make_named_tuple_subclass("ColumnTuple")


def record_to_dict(record: RecordBase) -> dict[str, Any]:
    """Converts an SQLAlchemy record to a dict."""
    return RecordDict(
        {col.key: getattr(record, col.key) for col in inspect(record).mapper.column_attrs}
    )


def _value_if_unique(x: tuple):
    # Assume that all members of the tuple are the same type, since they come from the same column
    if isinstance(x[0], list):
        x = tuple(map(tuple, x))  # type: ignore
    if len(set(x)) == 1:
        return x[0]
    else:
        return x


# TODO: Use a DataFrame instead
def records_to_dict(records: list[RecordBase], collapse_constant: bool = True) -> dict[str, Any]:
    """Zips multiple records into a single dict."""
    records_dict = jtree.zip_(
        *[record_to_dict(r) for r in records],
        is_leaf=allf(is_type(dict, list), is_not_type(RecordDict)),
        zip_cls=ColumnTuple,
    )
    if collapse_constant:
        records_dict = jt.map(
            lambda x: _value_if_unique(x),
            records_dict,
            is_leaf=is_type(ColumnTuple),
        )
    return records_dict


def retrieve_figures(
    session: Session,
    # model_parameters: Optional[dict] = None,
    evaluation_parameters: Optional[dict] = None,
    exclude_archived: bool = True,
    **figure_parameters,
) -> tuple[list[go.Figure], list[tuple[FigureRecord, EvaluationRecord, ModelRecord]]]:
    """Retrieve figures matching the given parameters.

    Parameters can contain tuples to match multiple values (OR condition).

    Args:
        session: Database session
        model_parameters: Parameters to match in models table
        evaluation_parameters: Parameters to match in evaluations table
        exclude_archived: If True, exclude archived figures
        **figure_parameters: Parameters to match in figures table

    Returns:
        Tuple of (list of plotly figures, list of record tuples)
        Each record tuple contains (figure_record, evaluation_record, model_record)
    """
    #! TODO: Update this to work with `model_hashes`
    # Start with base query joining all three tables
    query = (
        session.query(FigureRecord, EvaluationRecord, ModelRecord).join(
            EvaluationRecord, FigureRecord.evaluation_hash == EvaluationRecord.hash
        )
        # .join(ModelRecord, FigureRecord.model_hash == ModelRecord.hash)
    )

    if exclude_archived:
        query = query.filter(FigureRecord.archived == False)  # noqa: E712

    def add_filters(query, model, parameters):
        for param, value in parameters.items():
            if isinstance(value, tuple):
                query = query.filter(getattr(model, param).in_(value))
            else:
                query = query.filter(getattr(model, param) == value)
        return query

    # Add figure parameter filters
    query = add_filters(query, FigureRecord, figure_parameters)

    # Add model parameter filters
    # if model_parameters:
    #     query = add_filters(query, ModelRecord, model_parameters)

    # Add evaluation parameter filters
    if evaluation_parameters:
        query = add_filters(query, EvaluationRecord, evaluation_parameters)

    figures = []
    records = []
    for record_tuple in query.all():
        figure_record, _, _ = record_tuple
        json_path = PATHS.figures / figure_record.evaluation_hash / f"{figure_record.hash}.json"
        if json_path.exists():
            try:
                figures.append(plotly.io.read_json(json_path))
                records.append(record_tuple)
            except Exception as e:
                logger.warning(f"Failed to load figure {figure_record.hash}: {e}")

    # TODO: Report # of figures, associated with # of evaluations

    return figures, records


class _Box(eqx.Module):
    data: Any

    def unbox(self):
        return self.data


def record_to_hps_train(model_info: PyTree[ModelRecord]) -> TreeNamespace:
    """Convert database model record to training hyperparameters namespace.

    Extracts all training-related parameters from the database record and
    converts them back into a structured TreeNamespace suitable for model
    reconstruction.

    Args:
        model_info: PyTree of ModelRecord(s) from the database

    Returns:
        TreeNamespace containing training hyperparameters
    """
    sample_record: ModelRecord = jt.leaves(model_info, is_leaf=is_type(ModelRecord))[0]

    IGNORE_COLS = {
        "id",
        "hash",
        "expt_name",
        "created_at",
        "is_path_defunct",
        "version_info",
        "postprocessed",
        "has_replicate_info",
    }

    flat_params = {k: v for k, v in record_to_dict(sample_record).items() if k not in IGNORE_COLS}

    if not flat_params:
        raise ValueError("No training parameters found in model record")

    # The DB columns come from flattening the training hps namespace
    nested_params = unflatten_dict_keys(flat_params, sep=STRINGS.hps_level_label_sep)

    hps_train = dict_to_namespace(
        nested_params,
        to_type=TreeNamespace,
        exclude=is_dict_with_int_keys,
    )

    # Ensure special fields regain their wrapped types (e.g. where -> LDict)
    hps_train = cast_hps(hps_train, config_type="training")

    return hps_train


def fill_hps_with_train_params(
    hps: TreeNamespace,
    hps_train: TreeNamespace,
) -> TreeNamespace:
    """Fill analysis hyperparameters with training parameters.

    Merges training hyperparameters into analysis hyperparameters, replacing
    any None values in hps.train with values from hps_train. This is the
    replacement for fill_missing_train_hps_from_record that works with
    already-extracted training parameters.

    Args:
        hps: Analysis hyperparameters (may have None values under train)
        hps_train: Complete training hyperparameters from database

    Returns:
        Enriched analysis hyperparameters with train section filled
    """
    # Cast to ensure special fields regain their wrapped types (e.g. train.where -> LDict)
    return cast_hps(
        TreeNamespace(train=hps_train) | deepcopy(hps),
        config_type="analysis",
    )


def fill_missing_train_hps_from_record(
    hps: TreeNamespace,
    model_info: PyTree[ModelRecord],
) -> TreeNamespace:
    """Populate any *unspecified* values under ``train`` in *hps* with values taken
    from the first model record in *model_info* and return an **enriched copy**.

    Notes
    -----
    * All records in *model_info* are assumed to be identical except for
      ``train__pert__std`` (this is how training sweeps are organised in the
      current project).
    * Only columns whose names start with ``train__`` are considered.  A small
      set of bookkeeping columns (``id``, ``hash`` …) are always ignored.
    * The original *hps* namespace is **not** modified; the caller receives a
      new namespace instance containing the additional information.
    """

    # ---------------------------------------------------------------------
    # 1. Collect a representative DB record and turn it back into a nested
    #    namespace.
    # ---------------------------------------------------------------------
    sample_record: ModelRecord = jt.leaves(model_info, is_leaf=is_type(ModelRecord))[0]

    IGNORE_COLS = {
        "id",
        "hash",
        "expt_name",
        "created_at",
        "is_path_defunct",
        "version_info",
        "postprocessed",
        "has_replicate_info",
    }

    flat_params = {
        k: v
        for k, v in record_to_dict(sample_record).items()
        if (
            k not in IGNORE_COLS and k != "pert__std"  # varies across loaded models
        )
    }

    if not flat_params:
        # Nothing to add – return early
        return hps

    # The DB columns come from flattening the *training* hps namespace, so they
    # correspond to the subtree that should live under ``train``.
    nested_params = unflatten_dict_keys(flat_params, sep=STRINGS.hps_level_label_sep)
    record_hps_dict = {"train": nested_params}

    record_hps = dict_to_namespace(
        record_hps_dict,
        to_type=TreeNamespace,
        exclude=is_dict_with_int_keys,
    )

    # ---------------------------------------------------------------------
    # 2. Recursive merge: only replace values that are currently *None* or
    #    missing in *hps* so that explicit analysis-config overrides win.
    # ---------------------------------------------------------------------
    # Work on a *copy* so that callers can decide whether to keep or discard the
    # original object.
    new_hps = deepcopy(hps)

    def _merge_missing(dest: TreeNamespace, src: TreeNamespace):
        for attr, src_val in src.items():
            if hasattr(dest, attr):
                dest_val = getattr(dest, attr)
                if isinstance(dest_val, TreeNamespace) and isinstance(src_val, TreeNamespace):
                    _merge_missing(dest_val, src_val)
                elif dest_val is None:
                    setattr(dest, attr, src_val)
            else:
                setattr(dest, attr, src_val)

    _merge_missing(new_hps, record_hps)

    # Ensure special fields regain their wrapped types (e.g. train.where -> LDict)
    new_hps = cast_hps(new_hps, config_type="analysis")

    return new_hps


# def record_to_namespace(record: RecordBase, split_by='__') -> TreeNamespace:
#     """Convert an SQLAlchemy record to a TreeNamespace with nested structure.

#     Column names with underscores are converted to nested attributes.
#     For example, 'foo_bar_baz' becomes foo.bar.baz in the namespace.
#     """
#     # Get all column values as a dictionary
#     record_dict = {
#         # Wrap in `_Box` so that when we convert the dict to namespace, any dict-valued keys don't get converted
#         column_name: _Box(getattr(record, column_name))
#         for column_name in record.__table__.columns.keys()
#     }

#     record_key_paths = [
#         list(column_name.split(split_by)) for column_name in record_dict.keys()
#     ]

#     # Unflatten dict
#     hps_dict = None

#     # Convert to `TreeNamespace`

#     # Unbox values
