"""Streaming XES event log parser using lxml.

This module provides memory-efficient parsing of XES files by using
lxml's iterparse for streaming XML processing.
"""


import gzip
from datetime import datetime
from pathlib import Path
from typing import List, Union, Optional

from lxml import etree

from src.armature_light.models import Event, Trace

# XES namespace variants (some files use trailing slash, some don't)
XES_NS_VARIANTS = [
    "{http://www.xes-standard.org}",
    "{http://www.xes-standard.org/}",
]


def _strip_ns(tag: str) -> str:
    """Strip XES namespace prefix from tag."""
    for ns in XES_NS_VARIANTS:
        if tag.startswith(ns):
            return tag[len(ns) :]
    return tag


def parse_xes(path: Union[Path, str]) -> List[Trace]:
    """Parse XES event log into list of Trace objects.

    Uses streaming parser (iterparse) for memory efficiency on large files.
    Handles both uncompressed .xes and gzipped .xes.gz files.

    Args:
        path: Path to XES file (Path or string)

    Returns:
        List of Trace objects containing ordered events
    """
    path = Path(path)

    traces: List[Trace] = []
    current_trace_id: Optional[str] = None
    current_events: List[Event] = []
    trace_counter = 0

    # Open file handle - gzip for .gz, regular for .xes
    if str(path).endswith(".gz"):
        file_handle = gzip.open(path, "rb")
    else:
        file_handle = open(path, "rb")

    try:
        # Use iterparse for streaming - processes elements as they're encountered
        for event, elem in etree.iterparse(file_handle, events=("end",)):
            # Handle tags with or without namespace
            tag = _strip_ns(elem.tag)

            if tag == "trace":
                # End of trace - build Trace object
                if current_trace_id is None:
                    # Generate case_id if missing
                    current_trace_id = f"trace_{trace_counter}"
                    trace_counter += 1

                traces.append(Trace(case_id=current_trace_id, events=current_events))

                # Reset for next trace
                current_trace_id = None
                current_events = []
                elem.clear()  # Free memory

            elif tag == "event":
                # Extract activity and timestamp from child elements
                activity = None
                timestamp = None

                for child in elem:
                    child_tag = _strip_ns(child.tag)
                    if child_tag == "string" and child.get("key") == "concept:name":
                        activity = child.get("value")
                    elif child_tag == "date" and child.get("key") == "time:timestamp":
                        timestamp_str = child.get("value")
                        if timestamp_str:
                            timestamp = _parse_timestamp(timestamp_str)

                if activity:
                    current_events.append(Event(activity=activity, timestamp=timestamp))

                elem.clear()  # Free memory

            elif tag == "string" and elem.get("key") == "concept:name":
                # Check if this is trace-level concept:name (case_id)
                parent = elem.getparent()
                if parent is not None:
                    parent_tag = _strip_ns(parent.tag)
                    if parent_tag == "trace":
                        current_trace_id = elem.get("value")
    finally:
        file_handle.close()

    return traces


def _parse_timestamp(timestamp_str: str) -> datetime:
    """Parse XES timestamp string to datetime.

    XES timestamps are ISO 8601 format with timezone.
    Example: 2024-01-01T10:00:00.000+00:00

    Args:
        timestamp_str: ISO 8601 timestamp string

    Returns:
        Parsed datetime object
    """
    return datetime.fromisoformat(timestamp_str)
