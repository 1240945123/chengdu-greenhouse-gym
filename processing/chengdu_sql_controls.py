from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


TARGET_CONTROLLER_IDS = frozenset(range(43, 52))
SUPPORTED_ACTIONS = {"TURN_ON", "TURN_OFF", "STOP", "TURN_ON_2", "TURN_ON_4"}

OUTPUT_COLUMNS = [
    "log_id",
    "timestamp",
    "controller_id",
    "device",
    "action",
    "params",
    "source_content",
    "operator_id",
]

_TARGET_TABLES = {
    "equipment_controller_dev",
    "equipment_controller_operator_log",
}
_TABLE_RE = re.compile(r"^\s*INSERT\s+INTO\s+`?(?P<table>[A-Za-z0-9_]+)`?", re.IGNORECASE)
_VALUES_RE = re.compile(
    r"^\s*INSERT\s+INTO\s+`?(?P<table>[A-Za-z0-9_]+)`?\s+VALUES\s+(?P<values>.*)$",
    re.IGNORECASE,
)
_INSERT_RE = re.compile(
    r"^\s*INSERT\s+INTO\s+`?(?P<table>[A-Za-z0-9_]+)`?\s+VALUES\s+(?P<values>.+?)\s*;\s*$",
    re.IGNORECASE,
)
_MYSQL_ESCAPES = {
    "0": "\0",
    "b": "\b",
    "n": "\n",
    "r": "\r",
    "t": "\t",
    "Z": "\x1a",
}


def split_mysql_values(values_text: str) -> list[str | None]:
    """Parse one parenthesized MySQL VALUES row into decoded scalar strings."""
    return _split_mysql_values(values_text)


def _split_mysql_values(values_text: str, stop_after: int | None = None) -> list[str | None]:
    text = values_text.strip()
    if text.endswith(";"):
        text = text[:-1].rstrip()
    if text.startswith("("):
        if text.endswith(")"):
            text = text[1:-1]
        elif stop_after is None:
            raise ValueError("unclosed VALUES row")
        else:
            text = text[1:]

    values: list[str | None] = []
    token: list[str] = []
    quote: str | None = None
    field_was_quoted = False
    quote_closed = False
    index = 0

    def finish_field() -> None:
        nonlocal field_was_quoted, quote_closed
        value = "".join(token) if field_was_quoted else "".join(token).strip()
        if not value and not field_was_quoted:
            raise ValueError("empty value in VALUES row")
        values.append(None if not field_was_quoted and value.upper() == "NULL" else value)
        token.clear()
        field_was_quoted = False
        quote_closed = False

    while index < len(text):
        char = text[index]
        if quote is not None:
            if char == "\\":
                index += 1
                if index >= len(text):
                    raise ValueError("trailing backslash in quoted value")
                escaped = text[index]
                token.append(_MYSQL_ESCAPES.get(escaped, escaped))
            elif char == quote:
                if index + 1 < len(text) and text[index + 1] == quote:
                    token.append(quote)
                    index += 1
                else:
                    quote = None
                    quote_closed = True
            else:
                token.append(char)
        elif char in {"'", '"'}:
            if quote_closed or "".join(token).strip():
                raise ValueError("unexpected quote in unquoted value")
            token.clear()
            quote = char
            field_was_quoted = True
        elif char == ",":
            finish_field()
            if stop_after is not None and len(values) == stop_after:
                return values
        elif char in {"(", ")"}:
            raise ValueError("multiple or nested VALUES rows are not supported")
        elif quote_closed:
            if not char.isspace():
                raise ValueError("unexpected characters after quoted value")
        else:
            token.append(char)
        index += 1

    if quote is not None:
        raise ValueError("unterminated quoted value")
    finish_field()
    if stop_after is not None and len(values) < stop_after:
        raise ValueError(f"VALUES row has only {len(values)} fields; expected at least {stop_after}")
    return values


def extract_control_events(sql_path: str | Path) -> pd.DataFrame:
    """Extract target greenhouse controller events from a MySQL backup."""
    controller_names: dict[int, str] = {}
    event_rows: list[dict[str, object]] = []

    with Path(sql_path).open("r", encoding="utf-8-sig") as sql_file:
        for line_number, line in enumerate(sql_file, start=1):
            table_match = _TABLE_RE.match(line)
            if table_match is None:
                continue
            table = table_match.group("table").lower()
            if table not in _TARGET_TABLES:
                continue

            values_match = _VALUES_RE.match(line)
            if values_match is None:
                raise ValueError(f"line {line_number}: malformed INSERT for {table}")

            values_text = values_match.group("values")
            id_position = 1 if table == "equipment_controller_dev" else 7
            try:
                leading_values = _split_mysql_values(values_text, stop_after=id_position)
                controller_id = _integer(leading_values[-1], "controller ID")
            except ValueError:
                controller_id = None
            if controller_id is not None and controller_id not in TARGET_CONTROLLER_IDS:
                continue

            insert_match = _INSERT_RE.match(line)
            if insert_match is None:
                raise ValueError(f"line {line_number}: malformed INSERT for {table}")
            try:
                values_text = insert_match.group("values")
                values = split_mysql_values(values_text)
                if table == "equipment_controller_dev":
                    _collect_controller(values, line_number, controller_names)
                else:
                    event = _parse_event(values, line_number)
                    if event is not None:
                        event_rows.append(event)
            except ValueError as exc:
                if str(exc).startswith("line "):
                    raise
                raise ValueError(f"line {line_number}: {exc}") from exc

    for event in event_rows:
        controller_id = int(event["controller_id"])
        if controller_id not in controller_names:
            raise ValueError(
                f"line {event.pop('_line_number')}: missing controller definition for ID {controller_id}"
            )
        event["device"] = controller_names[controller_id]
        event.pop("_line_number")

    return _event_frame(event_rows)


def _collect_controller(values: list[str | None], line_number: int, names: dict[int, str]) -> None:
    if not values:
        raise ValueError("controller row has no values")
    controller_id = _integer(values[0], "controller ID")
    if controller_id not in TARGET_CONTROLLER_IDS:
        return
    if len(values) != 28:
        raise ValueError(f"target controller row has {len(values)} values; expected 28")
    device = values[9] or values[10]
    if not device:
        raise ValueError("target controller row has no device type or name")
    names[controller_id] = device


def _parse_event(values: list[str | None], line_number: int) -> dict[str, object] | None:
    if len(values) < 7:
        raise ValueError("operator-log row does not include a controller ID")
    controller_id = _integer(values[6], "controller ID")
    if controller_id not in TARGET_CONTROLLER_IDS:
        return None
    if len(values) != 11:
        raise ValueError(f"target operator-log row has {len(values)} values; expected 11")

    action = values[7]
    if action not in SUPPORTED_ACTIONS:
        raise ValueError(f"line {line_number}: unsupported target action {action!r}")
    if values[1] is None:
        raise ValueError("target operator-log row has no timestamp")
    try:
        timestamp = pd.Timestamp(values[1])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid target timestamp {values[1]!r}") from exc
    if pd.isna(timestamp):
        raise ValueError(f"invalid target timestamp {values[1]!r}")

    return {
        "log_id": _integer(values[0], "log ID"),
        "timestamp": timestamp,
        "controller_id": controller_id,
        "device": None,
        "action": action,
        "params": values[8],
        "source_content": values[9],
        "operator_id": _integer(values[10], "operator ID", nullable=True),
        "_line_number": line_number,
    }


def _integer(value: str | None, label: str, nullable: bool = False) -> int | None:
    if value is None:
        if nullable:
            return None
        raise ValueError(f"{label} is NULL")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {label}: {value!r}") from exc


def _event_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    if frame.empty:
        frame = frame.astype(
            {
                "log_id": "int64",
                "timestamp": "datetime64[ns]",
                "controller_id": "int64",
                "operator_id": "Int64",
            }
        )
        return frame

    frame["log_id"] = frame["log_id"].astype("int64")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame["controller_id"] = frame["controller_id"].astype("int64")
    frame["operator_id"] = pd.array(frame["operator_id"], dtype="Int64")
    return frame.sort_values(["timestamp", "log_id"], kind="stable").reset_index(drop=True)
