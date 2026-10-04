import os

from datetime import datetime

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    send_from_directory
)

from werkzeug.utils import secure_filename

from database import (
    initialize_database,
    create_video,
    update_video_status,
    search_detections,
    get_statistics,
    connection
)

from ai_engine import process_video


# =========================================================
# FLASK
# =========================================================

app = Flask(
    __name__
)


# =========================================================
# CONFIG
# =========================================================

UPLOAD_FOLDER = "uploads"

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

os.makedirs(
    "processed",
    exist_ok=True
)

os.makedirs(
    "snapshots",
    exist_ok=True
)


app.config[
    "MAX_CONTENT_LENGTH"
] = 1024 * 1024 * 1024


ALLOWED_EXTENSIONS = {
    "mp4",
    "avi",
    "mov",
    "mkv",
    "webm"
}


# =========================================================
# DATABASE
# =========================================================

initialize_database()


# =========================================================
# FILE CHECK
# =========================================================

def allowed_file(filename):

    return (
        "." in filename
        and
        filename.rsplit(
            ".",
            1
        )[1].lower()
        in ALLOWED_EXTENSIONS
    )


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():

    conn = connection()

    videos = conn.execute(
        """
        SELECT *
        FROM videos
        ORDER BY id DESC
        """
    ).fetchall()

    conn.close()

    return render_template(
        "index.html",
        page="home",
        videos=videos
    )


# =========================================================
# UPLOAD
# =========================================================

@app.route(
    "/upload",
    methods=["POST"]
)
def upload():

    if "video" not in request.files:

        return redirect(
            url_for("home")
        )

    file = request.files[
        "video"
    ]

    if file.filename == "":

        return redirect(
            url_for("home")
        )

    if not allowed_file(
        file.filename
    ):

        return (
            "Unsupported video format",
            400
        )

    original_name = secure_filename(
        file.filename
    )

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    video_id = create_video(
        original_name,
        timestamp
    )

    filename = (
        f"{video_id}_"
        f"{original_name}"
    )

    input_path = os.path.join(
        UPLOAD_FOLDER,
        filename
    )

    file.save(
        input_path
    )

    # =========================================
    # PROCESS
    #
    # Simple version processes immediately.
    # The browser waits until processing ends.
    # =========================================

    process_video(
        video_id,
        input_path
    )

    return redirect(
        url_for(
            "results",
            video_id=video_id
        )
    )


# =========================================================
# RESULTS
# =========================================================

@app.route(
    "/results/<int:video_id>"
)
def results(video_id):

    query = request.args.get(
        "q",
        ""
    ).strip()

    object_type = request.args.get(
        "type",
        ""
    ).strip()

    detections = search_detections(

        video_id,

        query=query,

        object_type=object_type
    )

    statistics = get_statistics(
        video_id
    )

    conn = connection()

    video = conn.execute(
        """
        SELECT *
        FROM videos
        WHERE id = ?
        """,
        (video_id,)
    ).fetchone()

    conn.close()

    if video is None:

        return (
            "Video not found",
            404
        )

    return render_template(
        "index.html",

        page="results",

        video=video,

        detections=detections,

        statistics=statistics,

        query=query,

        object_type=object_type
    )


# =========================================================
# PROCESSED VIDEO
# =========================================================

@app.route(
    "/processed/<filename>"
)
def processed_video(filename):

    return send_from_directory(
        "processed",
        filename
    )


# =========================================================
# SNAPSHOT
# =========================================================

@app.route(
    "/snapshots/<filename>"
)
def snapshot(filename):

    return send_from_directory(
        "snapshots",
        filename
    )


# =========================================================
# START
# =========================================================

if __name__ == "__main__":

    print()
    print(
        "================================"
    )

    print(
        "       AI CCTV SYSTEM"
    )

    print(
        "================================"
    )

    print(
        "Open:"
    )

    print(
        "http://127.0.0.1:5000"
    )

    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
