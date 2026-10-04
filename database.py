"""
database.py - SQLite persistence layer for visitors and events.

Provides ACID-compliant, thread-safe, and crash-resilient storage for unique visitors
and their entry/exit tracking events.
"""
import sqlite3
import os
from typing import Optional, List, Dict, Any


def get_connection(db_path: str) -> sqlite3.Connection:
    """
    Creates and returns a thread-safe SQLite connection configured with:
    - Write-Ahead Logging (WAL) for high concurrency and crash durability
    - 30-second busy timeout to prevent database lock contention
    - Foreign key constraints enabled
    - Row factory for dictionary-like column access
    """
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_db(db_path: str):
    """
    Initializes the database schema if tables do not already exist.
    Creates visitors table, events table, and query indexes.
    """
    with get_connection(db_path) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS visitors (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                face_id     TEXT    NOT NULL UNIQUE,
                first_seen  TEXT    NOT NULL,
                last_seen   TEXT    NOT NULL,
                visit_count INTEGER DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                face_id     TEXT    NOT NULL,
                event_type  TEXT    NOT NULL,
                timestamp   TEXT    NOT NULL,
                image_path  TEXT,
                FOREIGN KEY (face_id) REFERENCES visitors(face_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_events_face_id ON events(face_id);
            CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
            CREATE INDEX IF NOT EXISTS idx_visitors_face_id ON visitors(face_id);
        """)
        conn.commit()


def register_visitor(db_path: str, face_id: str, timestamp: str):
    """
    Registers a newly identified visitor.
    Uses INSERT OR IGNORE to guarantee idempotency and avoid duplicates.
    """
    with get_connection(db_path) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO visitors (face_id, first_seen, last_seen, visit_count) VALUES (?, ?, ?, 1)",
            (face_id, timestamp, timestamp),
        )
        conn.commit()


def update_visitor_last_seen(db_path: str, face_id: str, timestamp: str):
    """
    Updates the last seen timestamp and increments visit count for an existing visitor.
    """
    with get_connection(db_path) as conn:
        conn.execute(
            "UPDATE visitors SET last_seen = ?, visit_count = visit_count + 1 WHERE face_id = ?",
            (timestamp, face_id),
        )
        conn.commit()


def log_event(db_path: str, face_id: str, event_type: str, timestamp: str, image_path: str):
    """
    Records an entry or exit event associated with a specific face_id.
    """
    with get_connection(db_path) as conn:
        conn.execute(
            "INSERT INTO events (face_id, event_type, timestamp, image_path) VALUES (?, ?, ?, ?)",
            (face_id, event_type, timestamp, image_path),
        )
        conn.commit()


def get_unique_visitor_count(db_path: str) -> int:
    """Returns the total number of unique visitors registered in the database."""
    with get_connection(db_path) as conn:
        row = conn.execute("SELECT COUNT(*) FROM visitors").fetchone()
        return row[0] if row else 0


def get_all_visitors(db_path: str) -> List[Dict[str, Any]]:
    """Returns a list of all visitors ordered by first_seen timestamp."""
    with get_connection(db_path) as conn:
        rows = conn.execute("SELECT * FROM visitors ORDER BY first_seen DESC").fetchall()
        return [dict(r) for r in rows]


def visitor_exists(db_path: str, face_id: str) -> bool:
    """Checks whether a visitor with given face_id is registered in the database."""
    with get_connection(db_path) as conn:
        row = conn.execute("SELECT 1 FROM visitors WHERE face_id = ?", (face_id,)).fetchone()
        return row is not None


def get_recent_events(db_path: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Fetches the most recent entry/exit events."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_events_by_face(db_path: str, face_id: str) -> List[Dict[str, Any]]:
    """Fetches all events associated with a specific face_id."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM events WHERE face_id = ? ORDER BY id ASC", (face_id,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_statistics(db_path: str) -> Dict[str, Any]:
    """Computes comprehensive metrics from the database."""
    with get_connection(db_path) as conn:
        total_visitors = conn.execute("SELECT COUNT(*) FROM visitors").fetchone()[0]
        total_entries = conn.execute("SELECT COUNT(*) FROM events WHERE event_type = 'entry'").fetchone()[0]
        total_exits = conn.execute("SELECT COUNT(*) FROM events WHERE event_type = 'exit'").fetchone()[0]
        repeat_visitors = conn.execute("SELECT COUNT(*) FROM visitors WHERE visit_count > 1").fetchone()[0]
        return {
            "total_visitors": total_visitors,
            "total_entries": total_entries,
            "total_exits": total_exits,
            "repeat_visitors": repeat_visitors,
        }
