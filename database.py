import sqlite3
from pathlib import Path


DATABASE = Path("data/cctv.db")


def connection():
    DATABASE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    conn = sqlite3.connect(
        DATABASE
    )

    conn.row_factory = sqlite3.Row

    return conn


def initialize_database():

    conn = connection()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            uploaded_at TEXT NOT NULL,
            status TEXT NOT NULL
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            video_id INTEGER,

            timestamp TEXT,
            frame_number INTEGER,

            track_id INTEGER,

            object_type TEXT,

            confidence REAL,

            plate TEXT,

            snapshot TEXT,

            x1 INTEGER,
            y1 INTEGER,
            x2 INTEGER,
            y2 INTEGER,

            FOREIGN KEY(video_id)
                REFERENCES videos(id)
        )
    """)

    conn.commit()

    conn.close()


def create_video(
    filename,
    uploaded_at
):

    conn = connection()

    cursor = conn.execute(
        """
        INSERT INTO videos (
            filename,
            uploaded_at,
            status
        )
        VALUES (?, ?, ?)
        """,
        (
            filename,
            uploaded_at,
            "processing"
        )
    )

    video_id = cursor.lastrowid

    conn.commit()

    conn.close()

    return video_id


def update_video_status(
    video_id,
    status
):

    conn = connection()

    conn.execute(
        """
        UPDATE videos
        SET status = ?
        WHERE id = ?
        """,
        (
            status,
            video_id
        )
    )

    conn.commit()

    conn.close()


def add_detection(
    video_id,
    timestamp,
    frame_number,
    track_id,
    object_type,
    confidence,
    plate,
    snapshot,
    x1,
    y1,
    x2,
    y2
):

    conn = connection()

    conn.execute(
        """
        INSERT INTO detections (
            video_id,
            timestamp,
            frame_number,
            track_id,
            object_type,
            confidence,
            plate,
            snapshot,
            x1,
            y1,
            x2,
            y2
        )

        VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,
        (
            video_id,
            timestamp,
            frame_number,
            track_id,
            object_type,
            confidence,
            plate,
            snapshot,
            x1,
            y1,
            x2,
            y2
        )
    )

    conn.commit()

    conn.close()


def search_detections(
    video_id,
    query="",
    object_type=""
):

    conn = connection()

    sql = """
        SELECT *
        FROM detections
        WHERE video_id = ?
    """

    parameters = [
        video_id
    ]

    if query:

        sql += """
            AND (
                plate LIKE ?
                OR object_type LIKE ?
                OR CAST(track_id AS TEXT) LIKE ?
            )
        """

        search = f"%{query}%"

        parameters.extend([
            search,
            search,
            search
        ])

    if object_type:

        sql += """
            AND object_type = ?
        """

        parameters.append(
            object_type
        )

    sql += """
        ORDER BY id DESC
        LIMIT 500
    """

    rows = conn.execute(
        sql,
        parameters
    ).fetchall()

    conn.close()

    return rows


def get_statistics(
    video_id
):

    conn = connection()

    total = conn.execute(
        """
        SELECT COUNT(*) AS total
        FROM detections
        WHERE video_id = ?
        """,
        (video_id,)
    ).fetchone()["total"]

    people = conn.execute(
        """
        SELECT COUNT(DISTINCT track_id) AS total
        FROM detections
        WHERE video_id = ?
        AND object_type = 'person'
        """,
        (video_id,)
    ).fetchone()["total"]

    vehicles = conn.execute(
        """
        SELECT COUNT(DISTINCT track_id) AS total
        FROM detections
        WHERE video_id = ?

        AND object_type IN (
            'car',
            'motorcycle',
            'bus',
            'truck'
        )
        """,
        (video_id,)
    ).fetchone()["total"]

    plates = conn.execute(
        """
        SELECT COUNT(DISTINCT plate) AS total
        FROM detections
        WHERE video_id = ?
        AND plate != ''
        """,
        (video_id,)
    ).fetchone()["total"]

    conn.close()

    return {
        "total": total,
        "people": people,
        "vehicles": vehicles,
        "plates": plates
    }
