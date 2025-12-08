# tools/view_last_analysis.py
import json
import sqlite3
from pathlib import Path
from textwrap import shorten


DB_PATH = Path(__file__).resolve().parents[2] / "data" / "mnemosyne.db"


def to_hex_palette(palette_json: str | None):
    if not palette_json:
        return ""
    try:
        colors = json.loads(palette_json)
        return ", ".join("#{0:02x}{1:02x}{2:02x}".format(*c) for c in colors[:5])
    except Exception:
        return ""


def main(limit: int = 20):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    query = """
    SELECT
        ar.id,
        f.file_path,
        ar.caption,
        ar.tags,
        ar.objects,
        ar.mood,
        ar.is_sensitive,
        ar.contains_text,
        ar.aesthetic_score,
        ar.sharpness,
        ar.color_palette,
        ar.video_duration,
        ar.scene_changes,
        ar.latitude,
        ar.longitude,
        ar.altitude,
        ar.location_name,
        ar.analyzed_at,
        ar.analysis_version
    FROM analysis_results ar
    JOIN files f ON f.id = ar.file_id
    ORDER BY ar.id DESC
    LIMIT ?
    """

    for row in cur.execute(query, (limit,)):
        (
            ar_id,
            file_path,
            caption,
            tags_json,
            objects_json,
            mood,
            is_sensitive,
            contains_text,
            aesthetic_score,
            sharpness,
            palette_json,
            video_duration,
            scene_changes_json,
            latitude,
            longitude,
            altitude,
            location_name,
            analyzed_at,
            analysis_version,
        ) = row

        tags = ", ".join(json.loads(tags_json or "[]"))
        objects = ", ".join(json.loads(objects_json or "[]"))
        palette_hex = to_hex_palette(palette_json)

        print("=" * 80)
        print(f"Analysis ID : {ar_id}")
        print(f"File        : {file_path}")
        print(f"Analyzed At : {analyzed_at}")
        print(f"Version     : {analysis_version}")
        print(f"Caption     : {shorten(caption or '', width=70)}")
        print(f"Tags        : {tags}")
        print(f"Objects     : {objects}")
        print(f"Mood        : {mood} | Sensitive: {bool(is_sensitive)} | Contains Text: {bool(contains_text)}")
        print(f"Aesthetic   : {aesthetic_score:.4f} | Sharpness: {sharpness:.4f}")
        print(f"Palette     : {palette_hex}")

        # Optional fields
        if latitude or longitude:
            print(f"GPS         : lat={latitude}, lon={longitude}, alt={altitude}")
        if location_name:
            print(f"Location    : {location_name}")
        if scene_changes_json:
            print(f"SceneChanges: {scene_changes_json}")
        if video_duration:
            print(f"Video Dur   : {video_duration}s")

        print()

    conn.close()


if __name__ == "__main__":
    main()
