"""
Streamlit App for Human Evaluation of AI-Generated Videos
========================================================

Author
-------
Gaëtan Brison

Description
-------
This application presents users with a series of videos to classify as either
"AI-generated" or "Real". It stores the annotations locally and sends results to
a Google Sheet.

Modules
-------
- os, json, random, string: Standard Python libraries
- pathlib.Path: Path management
- datetime: Timestamps
- streamlit: UI rendering
- gspread, oauth2client: Google Sheets API access

Functions
---------
- send_to_google_sheet: Uploads annotations to Google Sheet.
- generate_user_id: Generates a unique user session ID.
- load_videos: Loads videos from a folder.
- sample_and_mix: Samples and shuffles real/fake videos.
- save_annotation: Saves JSON annotation to disk.

"""

import os
import json
import random
import string
from pathlib import Path
from datetime import datetime, timezone

import streamlit as st
import gspread
from oauth2client.service_account import ServiceAccountCredentials


def send_to_google_sheet(data_row, creds_path='gen-lang-client-0335099937-acfbd690ae96.json'):
    """Append a data row to a Google Sheet.

    Parameters
    ----------
    data_row : list
        A list of data values to append as a row.
    creds_path : str, optional
        Path to the Google service account credential JSON file.
    """
    creds_path = Path(__file__).parent / creds_path
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(str(creds_path), scope)
    client = gspread.authorize(creds)

    sheet_url = "https://docs.google.com/spreadsheets/d/1ZLpuGKtSwvZO0Ql734OgrWzRA9zprD-nOyemnFux4P0"
    sheet = client.open_by_url(sheet_url)
    sheet1 = sheet.sheet1

    header = ["User ID", "ID", "Video Path", "Name", "Ground Truth", "User Answer", "Comment", "Timestamp"]
    if not sheet1.get_all_values():
        sheet1.append_row(header, value_input_option='USER_ENTERED')
    sheet1.append_row(data_row, value_input_option='USER_ENTERED')


def generate_user_id(length=6):
    """Generate a random user ID.

    Parameters
    ----------
    length : int, optional
        Length of the ID string.

    Returns
    -------
    str
        Randomly generated user ID.
    """
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))


# Session State Initialization
if "user_id" not in st.session_state:
    st.session_state.user_id = generate_user_id()
if "video_index" not in st.session_state:
    st.session_state.video_index = 0
if "annotations" not in st.session_state:
    st.session_state.annotations = []
if "video_list" not in st.session_state:
    st.session_state.video_list = []
if "awaiting_explanation" not in st.session_state:
    st.session_state.awaiting_explanation = False
if "current_ann" not in st.session_state:
    st.session_state.current_ann = {}


def load_videos(folder, exts=(".mp4", ".avi", ".mkv")):
    """Load video file names from a given folder.

    Parameters
    ----------
    folder : str or Path
        Directory to scan for videos.
    exts : tuple of str, optional
        Valid video file extensions.

    Returns
    -------
    list
        List of video file names.
    """
    try:
        return sorted([f for f in os.listdir(folder) if f.lower().endswith(exts)])
    except Exception as e:
        st.error(f"Error reading {folder}: {e}")
        return []


def sample_and_mix(fake_folder, real_folder, n_each=10):
    """Sample and mix fake and real videos.

    Parameters
    ----------
    fake_folder : Path
        Folder containing fake videos.
    real_folder : Path
        Folder containing real videos.
    n_each : int, optional
        Number of videos to sample from each folder.

    Returns
    -------
    list of Path
        Shuffled list of sampled video paths.
    """
    fake = load_videos(fake_folder)
    real = load_videos(real_folder)
    pick_f = random.sample(fake, min(n_each, len(fake)))
    pick_r = random.sample(real, min(n_each, len(real)))
    paths = [Path(fake_folder)/v for v in pick_f] + [Path(real_folder)/v for v in pick_r]
    random.shuffle(paths)
    return paths


def save_annotation(ann, video_path, base="annotations"):
    """Save a single annotation to a JSON file.

    Parameters
    ----------
    ann : dict
        Annotation data.
    video_path : Path
        Path to the video file being annotated.
    base : str, optional
        Base folder to save annotations.
    """
    sub = video_path.parent.name
    outdir = Path(base)/sub
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir/(video_path.stem + ".json")
    try:
        with open(outfile, "w", encoding="utf-8") as f:
            json.dump(ann, f, indent=4)
    except Exception as e:
        st.warning(f"Could not save annotation for {outfile.name}: {e}")


# ---------------- UI START ----------------
st.title("AI-Generated Video Detection")
st.markdown(f"**🆔 Your session ID:** `{st.session_state.user_id}`")

if not st.session_state.video_list:
    base_dir = Path(__file__).parent / "random"
    fake_dir = base_dir / "Fake"
    real_dir = base_dir / "Real"
    if not fake_dir.is_dir() or not real_dir.is_dir():
        st.error("Ensure folders 'random/Fake' and 'random/Real' exist next to this script.")
        st.stop()
    st.session_state.video_list = sample_and_mix(fake_dir, real_dir, n_each=10)

total = len(st.session_state.video_list)

if st.session_state.video_index >= total:
    st.write("### 🎉 All videos done. Thank you!")
    master = Path("annotations") / "all_annotations.json"
    master.parent.mkdir(exist_ok=True)
    with open(master, "w", encoding="utf-8") as f:
        json.dump(st.session_state.annotations, f, indent=4)

    correct = sum(
        1
        for ann in st.session_state.annotations
        if ann["ai_generated"] == (ann["ground_truth"] == "Fake")
    )
    st.markdown(f"## You classified **{correct}/{total}** videos correctly!")
    st.balloons()
    st.stop()

idx = st.session_state.video_index
video_path = st.session_state.video_list[idx]

st.markdown(
    f"""
    <div style='padding: 0.75em 1.5em; background-color: #f0f2f6;
        border-left: 6px solid #4A90E2; border-radius: 8px;
        margin-bottom: 1em; font-size: 1.2em; font-weight: bold;'>
        🎬 Video <span style='color:#4A90E2'>{idx+1}</span> of <span style='color:#333'>{total}</span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.video(str(video_path))

if st.session_state.awaiting_explanation:
    st.subheader("Tell us why you chose that:")
    reason = st.text_area("Your explanation:", key=f"reason_{idx}")
    if st.button("Next Video", key=f"next_{idx}"):
        ann = st.session_state.current_ann
        ann["explanation"] = reason
        ann["timestamp"] = datetime.now(timezone.utc).isoformat()

        save_annotation(ann, video_path)
        st.session_state.annotations.append(ann)

        row = [
            st.session_state.user_id,
            idx + 1,
            str(video_path),
            video_path.name,
            video_path.parent.name,
            "Yes" if ann["ai_generated"] else "No",
            reason,
            ann["timestamp"],
        ]
        try:
            send_to_google_sheet(row)
        except Exception as e:
            st.error(f"❌ Error writing to Google Sheet: {e}")

        st.session_state.awaiting_explanation = False
        st.session_state.current_ann = {}
        st.session_state.video_index += 1
        st.rerun()
    st.stop()

st.subheader("Is this video AI-generated?")
col1, col2 = st.columns(2)

if col1.button("Yes", key=f"yes_{idx}"):
    st.session_state.current_ann = {
        "video": video_path.name,
        "ground_truth": video_path.parent.name,
        "ai_generated": True
    }
    st.session_state.awaiting_explanation = True
    st.rerun()

if col2.button("No", key=f"no_{idx}"):
    st.session_state.current_ann = {
        "video": video_path.name,
        "ground_truth": video_path.parent.name,
        "ai_generated": False
    }
    st.session_state.awaiting_explanation = True
    st.rerun()
