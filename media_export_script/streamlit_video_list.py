"""
Usage:
    streamlit run streamlit_video_list.py

Remote usage on cluster:
    streamlit run streamlit_video_list.py \
        --server.address 0.0.0.0 \
        --server.port 8501 \
        --server.headless true
"""


import streamlit as st
import os
import re
from pathlib import Path
import base64

# Set wide layout to maximize available width
st.set_page_config(page_title="Media Grid Viewer", layout="wide")


def list_media_files(root_dir: str, exts: list) -> list:
    """
    Recursively list media files in root_dir matching given extensions.
    """
    media_files = []
    root_path = Path(root_dir)
    if not root_path.exists():
        return media_files
    for ext in exts:
        for file in root_path.rglob(f"*{ext}"):
            media_files.append(file)
    return sorted(media_files)


# Sidebar controls
st.sidebar.title("Settings")
# Check if we're on macOS or the cluster
if os.path.exists("/Users/binxuwang"):
    # Path for macOS
    root_path = "/Users/binxuwang/Library/CloudStorage/OneDrive-HarvardUniversity/VVS_Accentuation_project/Figures/red_20250428-30_accentuation"
else:
    # Path for cluster check username is binxuwang
    # Path for cluster - check if username is binxuwang
    import getpass
    username = getpass.getuser()
    if username != "binxuwang":
        st.sidebar.warning(f"Current user is {username}, not binxuwang. Paths may need adjustment.")
    root_path = "/n/holylabs/LABS/alvarez_lab/Everyone/Accentuate_VVS/accentuation_outputs/"
root_dir = st.sidebar.text_input("Root directory", value=root_path)
regex_pattern = st.sidebar.text_input("Filter filenames (regex)", value="unit_0")
parent_folder_pattern = st.sidebar.text_input("Filter parent folder (regex)", value="")

# Compile regex, handle errors
regex = None
if regex_pattern:
    try:
        regex = re.compile(regex_pattern, re.IGNORECASE)
    except re.error as e:
        st.sidebar.error(f"Invalid filename regex: {e}")

parent_regex = None
if parent_folder_pattern:
    try:
        parent_regex = re.compile(parent_folder_pattern, re.IGNORECASE)
    except re.error as e:
        st.sidebar.error(f"Invalid parent folder regex: {e}")

extensions = st.sidebar.multiselect(
    "Select file types to display",
    options=[".gif", ".mp4", ".png", ".jpg"],
    default=[".gif", ]
)
cols = st.sidebar.slider("Number of columns", min_value=1, max_value=10, value=5)
video_width = st.sidebar.slider("Video width (px, HTML embed)", min_value=50, max_value=800, value=200)

# Main
st.title("Media Grid Viewer")
st.write("Displaying media files from:", root_dir)

# List and filter files
media_files = list_media_files(root_dir, extensions)
if regex:
    media_files = [f for f in media_files if regex.search(f.name)]
if parent_regex:
    media_files = [f for f in media_files if parent_regex.search(str(f.parent.name))]

if not media_files:
    st.warning("No media files found. Adjust your settings in the sidebar.")
else:
    # Build grid
    rows = [media_files[i : i + cols] for i in range(0, len(media_files), cols)]
    for row_files in rows:
        cols_streamlit = st.columns(cols)
        for file, col in zip(row_files, cols_streamlit):
            # Create caption with parent folder name and file name
            caption = f"{file.parent.name}\n{file.name}"
            if file.suffix.lower() == ".mp4":
                # Embed HTML video for precise width control
                video_bytes = file.read_bytes()
                b64 = base64.b64encode(video_bytes).decode()
                html = f"<video controls width='{video_width}'><source src='data:video/mp4;base64,{b64}' type='video/mp4'></video>"
                col.markdown(html, unsafe_allow_html=True)
                col.caption(caption)  # Add caption separately for videos
            else:
                # Images and GIFs use container width
                col.image(str(file), caption=caption, use_container_width=True)

st.write(f"Displayed {len(media_files)} items in a {cols}-column grid.")
