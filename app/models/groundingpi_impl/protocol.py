"""Canonical special-token format for GAM spatial supervision.

The module is deliberately dependency-free.  It owns the textual contract used
by data cleaning, cache validation, and evaluation; callers should not recreate
the grammar with ad-hoc regular expressions.

Target grammar (``task`` determines coordinate arity)::

    answer  := entry (", " entry)*
    entry   := REF_START phrase REF_END BOX_START payload BOX_END
    payload := "None" | tuple ("," tuple)*
    tuple   := coord coord                    # point
             | coord coord coord coord        # bbox
    coord   := "<0>" | ... | "<999>"

Phrases may contain natural-language commas and quotes.  They may not contain
any reserved token literal, because that would make the serialized stream
ambiguous or inject control tokens into training data.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
import unicodedata
from typing import Iterable, Optional, Sequence, Tuple, Union

GRID_SIZE = 1000
GRID_MAX = GRID_SIZE - 1

OBJECT_REF_START = "<|object_ref_start|>"
OBJECT_REF_END = "<|object_ref_end|>"
BOX_START = "<|box_start|>"
BOX_END = "<|box_end|>"
SEP_TOKEN = "</c>"
NEGATIVE_PAYLOAD = "None"

COORDINATE_TOKENS: Tuple[str, ...] = tuple(
    f"<{value}>" for value in range(GRID_SIZE)
)

DENSE_BBOX_PROMPT_PREFIX = (
    "Locate all the instances that match the following categories: "
)
REFER_BBOX_PROMPT_PREFIX = (
    "Locate the target referred to by the following description: "
)
DENSE_POINT_PROMPT_PREFIX = "Point to: "
REFER_POINT_PROMPT_PREFIX = (
    "Point to the target referred to by the following description: "
)
# Backward-compatible constant aliases.  New callers must use the explicit
# dense/refer names so task intent cannot be inferred from coordinate arity.
DETECTION_PROMPT_PREFIX = DENSE_BBOX_PROMPT_PREFIX
POINT_PROMPT_PREFIX = DENSE_POINT_PROMPT_PREFIX
VISUAL_PROMPT_PREFIX = "Given reference boxes "
VISUAL_PROMPT_SUFFIX = (
    " indicating one or more objects, find all similar objects in the image "
    "and output their bounding boxes."
)
VISUAL_PROMPT_LABEL = "object"
OCR_PROMPT = "OCR task detect all the text in box format."
OCR_PROMPT_PREFIX = "OCR task detect all the "
OCR_PROMPT_SUFFIX = " in box format."
OCR_UNITS = ("text line", "word")
LAYOUT_PROMPT_PREFIX = (
    "Detect all document layout elements that match the following categories: "
)
GUI_PROMPT_PREFIX = (
    "Point to the UI element to click for the following instruction: "
)
REASONING_INSTRUCTION = "Think step by step before giving the final answer."

_COORDINATE_SEPARATOR = ","
_ENTRY_SEPARATOR = ", "
_COORD_TOKEN_RE = re.compile(r"<([0-9]+)>")
_COORD_LITERAL_RE = re.compile(r"<[0-9]+>")
_QWEN_TOKEN_LITERAL_RE = re.compile(r"<\|[^<>]+\|>")

_LEGACY_OR_FOREIGN_TOKENS = (
    "<p>",
    "</p>",
    "<bbox>",
    "</bbox>",
    "<point>",
    "</point>",
    "<ref>",
    "</ref>",
    "<box>",
    "</box>",
    "<image>",
    "</image>",
)
_RESERVED_LITERALS = (
    OBJECT_REF_START,
    OBJECT_REF_END,
    BOX_START,
    BOX_END,
    SEP_TOKEN,
) + _LEGACY_OR_FOREIGN_TOKENS


class FormatError(ValueError):
    """Base class for special-token format errors."""


class PhraseSafetyError(FormatError):
    """Raised when a phrase is empty, non-canonical, or token-injecting."""


class CoordinateError(FormatError):
    """Raised when a coordinate tuple is malformed or outside its grid."""


class GrammarError(FormatError):
    """Raised when a serialized prompt or answer violates the full grammar."""


class TaskType(str, Enum):
    """Spatial target type; both types use Qwen's native box wrapper tokens."""

    BBOX = "bbox"
    POINT = "point"
    TRAJECTORY = "trajectory"

    @property
    def arity(self) -> int:
        return 4 if self is TaskType.BBOX else 2

    @property
    def preserves_coordinate_order(self) -> bool:
        """Trajectory points are temporal and must never be spatially sorted."""

        return self is TaskType.TRAJECTORY


class PromptFamily(str, Enum):
    """Semantic prompt intent; deliberately independent of geometry arity."""

    DENSE_BBOX = "dense_bbox"
    REFER_BBOX = "refer_bbox"
    DENSE_POINT = "dense_point"
    REFER_POINT = "refer_point"
    VISUAL_PROMPT = "visual_prompt"
    OCR = "ocr"
    LAYOUT = "layout"
    GUI = "gui"
    GENERAL_SUPPORT = "general_support"


class DataRoute(str, Enum):
    """Training routes; grounding and dense share one prompt."""

    REFERRING = "referring"
    GROUNDING = "grounding"
    DENSE = "dense"
    DENSE_POINT = "dense_point"
    REFER_POINT = "refer_point"
    GUI = "gui"
    OCR = "ocr"
    LAYOUT = "layout"
    VISUAL_PROMPT = "visual_prompt"
    GENERAL_SUPPORT = "general_support"


ROUTE_TO_PROMPT_FAMILY = {
    DataRoute.REFERRING: PromptFamily.REFER_BBOX,
    DataRoute.GROUNDING: PromptFamily.DENSE_BBOX,
    DataRoute.DENSE: PromptFamily.DENSE_BBOX,
    DataRoute.DENSE_POINT: PromptFamily.DENSE_POINT,
    DataRoute.REFER_POINT: PromptFamily.REFER_POINT,
    DataRoute.GUI: PromptFamily.GUI,
    DataRoute.OCR: PromptFamily.OCR,
    DataRoute.LAYOUT: PromptFamily.LAYOUT,
    DataRoute.VISUAL_PROMPT: PromptFamily.VISUAL_PROMPT,
    DataRoute.GENERAL_SUPPORT: PromptFamily.GENERAL_SUPPORT,
}


def route_prompt_family(route: Union[DataRoute, str]) -> PromptFamily:
    """Resolve one explicit data route to its canonical evaluation prompt."""

    try:
        data_route = (
            route if isinstance(route, DataRoute) else DataRoute(route)
        )
    except (TypeError, ValueError) as exc:
        raise FormatError(f"unsupported data route: {route!r}") from exc
    return ROUTE_TO_PROMPT_FAMILY[data_route]


CoordinateTuple = Tuple[int, ...]
CoordinateCollection = Tuple[CoordinateTuple, ...]


@dataclass(frozen=True)
class GroundingEntry:
    """One phrase and all of its boxes/points, or ``None`` for a negative."""

    phrase: str
    coordinates: Optional[CoordinateCollection]


def _coerce_task(task: Union[TaskType, str]) -> TaskType:
    if isinstance(task, TaskType):
        return task
    try:
        return TaskType(task)
    except (TypeError, ValueError) as exc:
        raise FormatError(f"unsupported task type: {task!r}") from exc


def coordinate_token(value: int) -> str:
    """Return the atomic token for one coordinate in the 0..999 grid."""

    _validate_scalar(value, GRID_MAX, "coordinate")
    return COORDINATE_TOKENS[value]


def canonicalize_phrase(phrase: str) -> str:
    """NFC-normalize and trim a phrase, then enforce token-injection safety.

    Internal punctuation and whitespace are preserved.  In particular, commas
    are legal because multiple queries are separated by the atomic ``</c>``
    token rather than natural-language punctuation.
    """

    if not isinstance(phrase, str):
        raise PhraseSafetyError(
            f"phrase must be str, got {type(phrase).__name__}"
        )

    normalized = unicodedata.normalize("NFC", phrase).strip()
    validate_phrase(normalized)
    return normalized


def validate_phrase(phrase: str, *, require_canonical: bool = True) -> None:
    """Validate a phrase without modifying it."""

    if not isinstance(phrase, str):
        raise PhraseSafetyError(
            f"phrase must be str, got {type(phrase).__name__}"
        )
    if not phrase:
        raise PhraseSafetyError("phrase must not be empty")
    if require_canonical:
        if phrase != phrase.strip():
            raise PhraseSafetyError(
                "phrase has leading or trailing whitespace"
            )
        if phrase != unicodedata.normalize("NFC", phrase):
            raise PhraseSafetyError("phrase is not NFC-normalized")

    for character in phrase:
        if unicodedata.category(character) in {"Cc", "Cf", "Cs"}:
            raise PhraseSafetyError(
                f"phrase contains unsafe Unicode control/format character "
                f"U+{ord(character):04X}"
            )

    for literal in _RESERVED_LITERALS:
        if literal in phrase:
            raise PhraseSafetyError(
                f"phrase contains reserved token {literal!r}"
            )
    if _COORD_LITERAL_RE.search(phrase):
        raise PhraseSafetyError("phrase contains a coordinate-token literal")
    if _QWEN_TOKEN_LITERAL_RE.search(phrase):
        raise PhraseSafetyError("phrase contains a Qwen control-token literal")


def _validate_scalar(value: int, grid_max: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CoordinateError(f"{label} must be an integer, got {value!r}")
    if not 0 <= value <= grid_max:
        raise CoordinateError(
            f"{label} {value} is outside the inclusive grid [0, {grid_max}]"
        )


def _validate_tuple(
    coordinate: CoordinateTuple,
    task: TaskType,
    *,
    grid_max: int,
) -> None:
    if len(coordinate) != task.arity:
        raise CoordinateError(
            f"{task.value} tuple needs {task.arity} coordinates, "
            f"got {len(coordinate)}: {coordinate!r}"
        )
    for value in coordinate:
        _validate_scalar(value, grid_max, "coordinate")

    if task is TaskType.BBOX:
        x1, y1, x2, y2 = coordinate
        if x1 >= x2 or y1 >= y2:
            raise CoordinateError(
                "bbox must be non-degenerate with x1 < x2 and y1 < y2: "
                f"{coordinate!r}"
            )


def _normalize_coordinates(
    coordinates: Sequence[Sequence[int]],
    task: TaskType,
    *,
    sort_coordinates: bool,
    require_sorted: bool = False,
) -> CoordinateCollection:
    if isinstance(coordinates, (str, bytes)):
        raise CoordinateError("coordinate collection must not be a string")
    try:
        normalized = tuple(tuple(item) for item in coordinates)
    except TypeError as exc:
        raise CoordinateError(
            "coordinates must be an iterable of tuples"
        ) from exc
    if not normalized:
        raise CoordinateError(
            "coordinate collection must not be empty; use None"
        )

    for coordinate in normalized:
        _validate_tuple(coordinate, task, grid_max=GRID_MAX)

    if not task.preserves_coordinate_order and len(set(normalized)) != len(
        normalized
    ):
        raise CoordinateError(
            "duplicate coordinate tuples must be removed explicitly"
        )
    if (
        require_sorted
        and not task.preserves_coordinate_order
        and normalized != tuple(sorted(normalized, key=lambda item: item[0]))
    ):
        raise CoordinateError(
            "coordinate tuples are not in canonical sorted order"
        )
    if sort_coordinates and not task.preserves_coordinate_order:
        # Rex-Omni sorts instances by the first coordinate only. Python's
        # stable sort preserves source annotation order when x/x1 is tied.
        normalized = tuple(sorted(normalized, key=lambda item: item[0]))
    return normalized


def _normalize_entries(
    entries: Iterable[GroundingEntry],
    task: TaskType,
    *,
    sort_coordinates: bool,
    require_sorted: bool = False,
) -> Tuple[GroundingEntry, ...]:
    try:
        materialized = tuple(entries)
    except TypeError as exc:
        raise FormatError("entries must be iterable") from exc
    if not materialized:
        raise FormatError("answer must contain at least one entry")

    normalized_entries = []
    seen_phrases = set()
    for index, entry in enumerate(materialized):
        if not isinstance(entry, GroundingEntry):
            raise FormatError(
                f"entry {index} must be GroundingEntry, got "
                f"{type(entry).__name__}"
            )
        phrase = canonicalize_phrase(entry.phrase)
        if phrase in seen_phrases:
            raise FormatError(f"duplicate phrase entry: {phrase!r}")
        seen_phrases.add(phrase)

        coordinates = entry.coordinates
        if coordinates is not None:
            coordinates = _normalize_coordinates(
                coordinates,
                task,
                sort_coordinates=sort_coordinates,
                require_sorted=require_sorted,
            )
        normalized_entries.append(GroundingEntry(phrase, coordinates))
    return tuple(normalized_entries)


def _serialize_payload(coordinates: CoordinateCollection) -> str:
    return _COORDINATE_SEPARATOR.join(
        "".join(coordinate_token(value) for value in coordinate)
        for coordinate in coordinates
    )


def serialize_answer(
    entries: Iterable[GroundingEntry],
    task: Union[TaskType, str],
) -> str:
    """Serialize entries into the unique canonical answer representation."""

    task_type = _coerce_task(task)
    normalized = _normalize_entries(
        entries,
        task_type,
        sort_coordinates=True,
    )
    chunks = []
    for entry in normalized:
        payload = (
            NEGATIVE_PAYLOAD
            if entry.coordinates is None
            else _serialize_payload(entry.coordinates)
        )
        chunks.append(
            f"{OBJECT_REF_START}{entry.phrase}{OBJECT_REF_END}"
            f"{BOX_START}{payload}{BOX_END}"
        )
    return _ENTRY_SEPARATOR.join(chunks)


def _expect(text: str, position: int, literal: str, context: str) -> int:
    if not text.startswith(literal, position):
        found = text[position : position + max(len(literal), 16)]
        raise GrammarError(
            f"expected {literal!r} at offset {position} ({context}); found {found!r}"
        )
    return position + len(literal)


def _parse_coordinate_token(payload: str, position: int) -> Tuple[int, int]:
    match = _COORD_TOKEN_RE.match(payload, position)
    if match is None:
        found = payload[position : position + 16]
        raise GrammarError(
            f"expected coordinate token at payload offset {position}; found {found!r}"
        )
    digits = match.group(1)
    value = int(digits)
    if digits != str(value):
        raise GrammarError(f"non-canonical coordinate token <{digits}>")
    _validate_scalar(value, GRID_MAX, "coordinate")
    return value, match.end()


def _parse_payload(
    payload: str,
    task: TaskType,
    *,
    allow_none: bool,
    require_canonical: bool,
) -> Optional[CoordinateCollection]:
    if payload == NEGATIVE_PAYLOAD:
        if not allow_none:
            raise GrammarError("None is not valid in this coordinate payload")
        return None
    if not payload:
        raise GrammarError("coordinate payload must not be empty")

    position = 0
    coordinates = []
    while position < len(payload):
        coordinate = []
        for _ in range(task.arity):
            value, position = _parse_coordinate_token(payload, position)
            coordinate.append(value)
        coordinate_tuple = tuple(coordinate)
        _validate_tuple(coordinate_tuple, task, grid_max=GRID_MAX)
        coordinates.append(coordinate_tuple)

        if position == len(payload):
            break
        position = _expect(
            payload,
            position,
            _COORDINATE_SEPARATOR,
            "coordinate tuple separator",
        )
        if position == len(payload):
            raise GrammarError("trailing coordinate tuple separator")

    normalized = tuple(coordinates)
    if not task.preserves_coordinate_order and len(set(normalized)) != len(
        normalized
    ):
        raise CoordinateError("duplicate coordinate tuples are not canonical")
    if (
        require_canonical
        and not task.preserves_coordinate_order
        and normalized != tuple(sorted(normalized, key=lambda item: item[0]))
    ):
        raise CoordinateError(
            "coordinate tuples are not in canonical sorted order"
        )
    return normalized


def parse_answer(
    text: str,
    task: Union[TaskType, str],
    *,
    require_canonical: bool = True,
) -> Tuple[GroundingEntry, ...]:
    """Parse an answer and require the grammar to consume the complete string."""

    if not isinstance(text, str):
        raise GrammarError(f"answer must be str, got {type(text).__name__}")
    if not text:
        raise GrammarError("answer must not be empty")
    task_type = _coerce_task(task)

    position = 0
    entries = []
    seen_phrases = set()
    while position < len(text):
        position = _expect(text, position, OBJECT_REF_START, "entry start")
        phrase_end = text.find(OBJECT_REF_END, position)
        if phrase_end < 0:
            raise GrammarError(
                f"missing {OBJECT_REF_END!r} for phrase beginning at offset {position}"
            )
        phrase = text[position:phrase_end]
        try:
            validate_phrase(phrase, require_canonical=require_canonical)
        except PhraseSafetyError as exc:
            raise GrammarError(
                f"unsafe phrase at offset {position}: {exc}"
            ) from exc
        if phrase in seen_phrases:
            raise GrammarError(f"duplicate phrase entry: {phrase!r}")
        seen_phrases.add(phrase)
        position = phrase_end + len(OBJECT_REF_END)

        position = _expect(text, position, BOX_START, "box payload start")
        payload_end = text.find(BOX_END, position)
        if payload_end < 0:
            raise GrammarError(
                f"missing {BOX_END!r} for payload beginning at offset {position}"
            )
        payload = text[position:payload_end]
        coordinates = _parse_payload(
            payload,
            task_type,
            allow_none=True,
            require_canonical=require_canonical,
        )
        entries.append(GroundingEntry(phrase, coordinates))
        position = payload_end + len(BOX_END)

        if position == len(text):
            break
        position = _expect(text, position, _ENTRY_SEPARATOR, "entry separator")
        if position == len(text):
            raise GrammarError("trailing entry separator")

    parsed = tuple(entries)
    if require_canonical and serialize_answer(parsed, task_type) != text:
        raise GrammarError(
            "answer is valid but not in canonical representation"
        )
    return parsed


def validate_answer(
    text: str,
    task: Union[TaskType, str],
    *,
    require_canonical: bool = True,
) -> None:
    """Validate an answer, raising a typed error on the first violation."""

    parse_answer(text, task, require_canonical=require_canonical)


def _normalize_prompt_phrases(phrases: Iterable[str]) -> Tuple[str, ...]:
    if isinstance(phrases, (str, bytes)):
        raise PhraseSafetyError("phrases must be a collection, not one string")
    try:
        normalized = tuple(canonicalize_phrase(phrase) for phrase in phrases)
    except TypeError as exc:
        raise PhraseSafetyError("phrases must be iterable") from exc
    if not normalized:
        raise PhraseSafetyError("prompt must contain at least one phrase")
    if len(set(normalized)) != len(normalized):
        raise PhraseSafetyError("prompt contains duplicate phrases")
    return normalized


def build_dense_bbox_prompt(phrases: Iterable[str]) -> str:
    """Build a category-recall bbox prompt using ``</c>`` between labels."""

    normalized = _normalize_prompt_phrases(phrases)
    return DENSE_BBOX_PROMPT_PREFIX + SEP_TOKEN.join(normalized) + "."


def parse_dense_bbox_prompt(text: str) -> Tuple[str, ...]:
    """Parse the exact prompt emitted by :func:`build_dense_bbox_prompt`."""

    if not isinstance(text, str) or not text.startswith(
        DENSE_BBOX_PROMPT_PREFIX
    ):
        raise GrammarError("invalid dense bbox prompt prefix")
    if not text.endswith("."):
        raise GrammarError("dense bbox prompt must end with '.'")
    joined = text[len(DENSE_BBOX_PROMPT_PREFIX) : -1]
    if not joined:
        raise GrammarError("dense bbox prompt contains no category")
    raw_phrases = tuple(joined.split(SEP_TOKEN))
    if any(not phrase for phrase in raw_phrases):
        raise GrammarError(
            "dense bbox prompt has an empty separator component"
        )
    try:
        normalized = _normalize_prompt_phrases(raw_phrases)
    except PhraseSafetyError as exc:
        raise GrammarError(
            f"unsafe dense bbox prompt category: {exc}"
        ) from exc
    if normalized != raw_phrases:
        raise GrammarError("dense bbox prompt categories are not canonical")
    if build_dense_bbox_prompt(normalized) != text:
        raise GrammarError("dense bbox prompt is not canonical")
    return normalized


def build_detection_prompt(phrases: Iterable[str]) -> str:
    """Compatibility alias for :func:`build_dense_bbox_prompt`."""

    return build_dense_bbox_prompt(phrases)


def parse_detection_prompt(text: str) -> Tuple[str, ...]:
    """Compatibility alias for :func:`parse_dense_bbox_prompt`."""

    return parse_dense_bbox_prompt(text)


def _build_refer_prompt(phrase: str, prefix: str) -> str:
    return prefix + canonicalize_phrase(phrase) + "."


def _parse_refer_prompt(text: str, prefix: str, task_label: str) -> str:
    if not isinstance(text, str) or not text.startswith(prefix):
        raise GrammarError(f"invalid {task_label} prompt prefix")
    if not text.endswith("."):
        raise GrammarError(f"{task_label} prompt must end with '.'")
    phrase = text[len(prefix) : -1]
    try:
        validate_phrase(phrase)
    except PhraseSafetyError as exc:
        raise GrammarError(
            f"unsafe {task_label} prompt phrase: {exc}"
        ) from exc
    if _build_refer_prompt(phrase, prefix) != text:
        raise GrammarError(f"{task_label} prompt is not canonical")
    return phrase


def build_refer_bbox_prompt(phrase: str) -> str:
    """Build a description-resolution bbox prompt with no count assumption."""

    return _build_refer_prompt(phrase, REFER_BBOX_PROMPT_PREFIX)


def parse_refer_bbox_prompt(text: str) -> str:
    """Parse the exact prompt emitted by :func:`build_refer_bbox_prompt`."""

    return _parse_refer_prompt(text, REFER_BBOX_PROMPT_PREFIX, "refer bbox")


def build_dense_point_prompt(phrases: Iterable[str]) -> str:
    """Build a category-recall point prompt using ``</c>`` between labels."""

    normalized = _normalize_prompt_phrases(phrases)
    return DENSE_POINT_PROMPT_PREFIX + SEP_TOKEN.join(normalized) + "."


def parse_dense_point_prompt(text: str) -> Tuple[str, ...]:
    """Parse the exact multi-category point prompt emitted by its builder."""

    if not isinstance(text, str) or not text.startswith(
        DENSE_POINT_PROMPT_PREFIX
    ):
        raise GrammarError("invalid dense point prompt prefix")
    if not text.endswith("."):
        raise GrammarError("dense point prompt must end with '.'")
    joined = text[len(DENSE_POINT_PROMPT_PREFIX) : -1]
    if not joined:
        raise GrammarError("dense point prompt contains no category")
    raw_phrases = tuple(joined.split(SEP_TOKEN))
    if any(not phrase for phrase in raw_phrases):
        raise GrammarError(
            "dense point prompt has an empty separator component"
        )
    try:
        normalized = _normalize_prompt_phrases(raw_phrases)
    except PhraseSafetyError as exc:
        raise GrammarError(
            f"unsafe dense point prompt category: {exc}"
        ) from exc
    if (
        normalized != raw_phrases
        or build_dense_point_prompt(normalized) != text
    ):
        raise GrammarError("dense point prompt is not canonical")
    return normalized


def build_point_prompt(phrase: str) -> str:
    """Compatibility builder for a single category-level point prompt."""

    return build_dense_point_prompt((phrase,))


def parse_point_prompt(text: str) -> str:
    """Parse the exact pointing prompt emitted by :func:`build_point_prompt`."""

    if not isinstance(text, str) or not text.startswith(
        DENSE_POINT_PROMPT_PREFIX
    ):
        raise GrammarError("invalid point prompt prefix")
    if not text.endswith("."):
        raise GrammarError("point prompt must end with '.'")
    phrases = parse_dense_point_prompt(text)
    if len(phrases) != 1:
        raise GrammarError("single point prompt contains multiple categories")
    return phrases[0]


def build_refer_point_prompt(phrase: str) -> str:
    """Build a description-resolution point prompt distinct from noun pointing."""

    return _build_refer_prompt(phrase, REFER_POINT_PROMPT_PREFIX)


def parse_refer_point_prompt(text: str) -> str:
    """Parse the exact prompt emitted by :func:`build_refer_point_prompt`."""

    return _parse_refer_prompt(text, REFER_POINT_PROMPT_PREFIX, "refer point")


def build_visual_prompt(reference_boxes: Sequence[Sequence[int]]) -> str:
    """Build the Rex-Omni-style visual-prompting instruction."""

    normalized = _normalize_coordinates(
        reference_boxes,
        TaskType.BBOX,
        sort_coordinates=True,
    )
    payload = _serialize_payload(normalized)
    return (
        VISUAL_PROMPT_PREFIX
        + BOX_START
        + payload
        + BOX_END
        + VISUAL_PROMPT_SUFFIX
    )


def parse_visual_prompt(text: str) -> CoordinateCollection:
    """Parse and fully validate a canonical visual-prompting instruction."""

    if not isinstance(text, str) or not text.startswith(VISUAL_PROMPT_PREFIX):
        raise GrammarError("invalid visual prompt prefix")
    if not text.endswith(VISUAL_PROMPT_SUFFIX):
        raise GrammarError("invalid visual prompt suffix")

    body = text[
        len(VISUAL_PROMPT_PREFIX) : len(text) - len(VISUAL_PROMPT_SUFFIX)
    ]
    if not body.startswith(BOX_START) or not body.endswith(BOX_END):
        raise GrammarError(
            "visual prompt must contain one wrapped bbox payload"
        )
    payload = body[len(BOX_START) : -len(BOX_END)]
    parsed = _parse_payload(
        payload,
        TaskType.BBOX,
        allow_none=False,
        require_canonical=True,
    )
    assert (
        parsed is not None
    )  # allow_none=False makes this an internal invariant.
    if build_visual_prompt(parsed) != text:
        raise GrammarError("visual prompt is not canonical")
    return parsed


def build_ocr_prompt(unit: str = "text") -> str:
    """Build the canonical scene-text spotting prompt for one annotation unit.

    OCR response labels are transcriptions discovered from the image, rather
    than query labels known when the prompt is built.  Each unique
    transcription therefore becomes one object-ref entry whose box wrapper
    contains all matching regions.
    """

    normalized = canonicalize_phrase(unit)
    if normalized not in OCR_UNITS and normalized != "text":
        raise PhraseSafetyError(
            f"unsupported OCR unit {unit!r}; expected text or one of {OCR_UNITS!r}"
        )
    return OCR_PROMPT


def parse_ocr_prompt(text: str) -> str:
    """Parse the exact prompt emitted by :func:`build_ocr_prompt`."""

    if text != OCR_PROMPT:
        raise GrammarError("OCR prompt is not canonical")
    return "text"


def build_layout_prompt(categories: Iterable[str]) -> str:
    """Build a document-layout category localization prompt."""

    normalized = _normalize_prompt_phrases(categories)
    return LAYOUT_PROMPT_PREFIX + SEP_TOKEN.join(normalized) + "."


def parse_layout_prompt(text: str) -> Tuple[str, ...]:
    """Parse the exact prompt emitted by :func:`build_layout_prompt`."""

    if not isinstance(text, str) or not text.startswith(LAYOUT_PROMPT_PREFIX):
        raise GrammarError("invalid layout prompt prefix")
    if not text.endswith("."):
        raise GrammarError("layout prompt must end with '.'")
    joined = text[len(LAYOUT_PROMPT_PREFIX) : -1]
    raw_categories = tuple(joined.split(SEP_TOKEN)) if joined else ()
    if not raw_categories or any(not category for category in raw_categories):
        raise GrammarError("layout prompt contains an empty category")
    try:
        normalized = _normalize_prompt_phrases(raw_categories)
    except PhraseSafetyError as exc:
        raise GrammarError(f"unsafe layout prompt category: {exc}") from exc
    if normalized != raw_categories or build_layout_prompt(normalized) != text:
        raise GrammarError("layout prompt is not canonical")
    return normalized


def build_gui_prompt(instruction: str) -> str:
    """Build a single-step GUI click-grounding prompt."""

    return _build_refer_prompt(instruction, GUI_PROMPT_PREFIX)


def parse_gui_prompt(text: str) -> str:
    """Parse the exact prompt emitted by :func:`build_gui_prompt`."""

    return _parse_refer_prompt(text, GUI_PROMPT_PREFIX, "GUI")


def remap_1000_coordinate(value: int) -> int:
    """Map one LocateAnything 0..1000 coordinate onto GAM's 0..999 grid.

    The implementation is exact integer round-half-up of ``value * 999 / 1000``;
    it does not use Python's bankers rounding and preserves both endpoints.
    """

    _validate_scalar(value, 1000, "source coordinate")
    return (value * GRID_MAX + 500) // 1000


def remap_1000_coordinates(
    coordinates: Sequence[Sequence[int]],
    task: Union[TaskType, str],
) -> CoordinateCollection:
    """Remap tuples from 0..1000 to 0..999 and reject post-map collapse."""

    task_type = _coerce_task(task)
    if isinstance(coordinates, (str, bytes)):
        raise CoordinateError("coordinate collection must not be a string")
    try:
        source = tuple(tuple(item) for item in coordinates)
    except TypeError as exc:
        raise CoordinateError(
            "coordinates must be an iterable of tuples"
        ) from exc
    if not source:
        raise CoordinateError("coordinate collection must not be empty")

    remapped = []
    for coordinate in source:
        _validate_tuple(coordinate, task_type, grid_max=1000)
        mapped = tuple(remap_1000_coordinate(value) for value in coordinate)
        _validate_tuple(mapped, task_type, grid_max=GRID_MAX)
        remapped.append(mapped)

    result = tuple(remapped)
    if not task_type.preserves_coordinate_order and len(set(result)) != len(
        result
    ):
        raise CoordinateError(
            "1000->999 remapping produced duplicate coordinate tuples"
        )
    return result


def validate_prompt_answer_pair(
    prompt: str,
    answer: str,
    family: Union[PromptFamily, str],
) -> Tuple[GroundingEntry, ...]:
    """Validate the complete GAM prompt/response contract.

    Every response entry must be exactly ``object_ref(label)`` immediately
    followed by one ``box(payload)`` wrapper.  ``parse_answer`` additionally
    guarantees that each label appears once and that the one payload contains
    zero/negative (``None``) or N canonical tuples.  Tuple arity is four for
    bbox families and two for point families.
    """

    try:
        prompt_family = (
            family
            if isinstance(family, PromptFamily)
            else PromptFamily(family)
        )
    except (TypeError, ValueError) as exc:
        raise FormatError(f"unsupported prompt family: {family!r}") from exc

    if prompt_family is PromptFamily.DENSE_BBOX:
        expected_phrases = parse_dense_bbox_prompt(prompt)
        task = TaskType.BBOX
    elif prompt_family is PromptFamily.REFER_BBOX:
        expected_phrases = (parse_refer_bbox_prompt(prompt),)
        task = TaskType.BBOX
    elif prompt_family is PromptFamily.DENSE_POINT:
        expected_phrases = parse_dense_point_prompt(prompt)
        task = TaskType.POINT
    elif prompt_family is PromptFamily.REFER_POINT:
        expected_phrases = (parse_refer_point_prompt(prompt),)
        task = TaskType.POINT
    elif prompt_family is PromptFamily.VISUAL_PROMPT:
        parse_visual_prompt(prompt)
        expected_phrases = (VISUAL_PROMPT_LABEL,)
        task = TaskType.BBOX
    elif prompt_family is PromptFamily.OCR:
        parse_ocr_prompt(prompt)
        # OCR transcriptions are discovered from the image and cannot be
        # enumerated in the prompt.  Full response grammar is still strict.
        expected_phrases = None
        task = TaskType.BBOX
    elif prompt_family is PromptFamily.LAYOUT:
        expected_phrases = parse_layout_prompt(prompt)
        task = TaskType.BBOX
    else:
        expected_phrases = (parse_gui_prompt(prompt),)
        task = TaskType.POINT

    entries = parse_answer(answer, task, require_canonical=True)
    actual_phrases = tuple(entry.phrase for entry in entries)
    if expected_phrases is not None and actual_phrases != expected_phrases:
        raise GrammarError(
            "prompt/answer label mismatch: "
            f"expected {expected_phrases!r}, got {actual_phrases!r}"
        )
    return entries
