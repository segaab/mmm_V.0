import os, json, time, zipfile, io, shutil
import streamlit as st
from kaggle.api.kaggle_api_extended import KaggleApi
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ---------- helpers ----------------------------------------------------------
@st.cache_resource
def get_kaggle_api():
    api = KaggleApi()
    api.authenticate()
    return api

def zip_dir(src: Path, dst_zip: Path):
    with zipfile.ZipFile(dst_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in src.rglob("*"):
            zf.write(p, p.relative_to(src))

def write_dataset_metadata(folder: Path, title, identifier):
    meta = {
        "title": title,
        "id": identifier,
        "licenses": [{"name": "CC0-1.0"}]
    }
    (folder/"dataset-metadata.json").write_text(json.dumps(meta, indent=2))

def write_kernel_metadata(folder: Path, title, identifier,
                          dataset_source, gpu: bool):
    meta = {
        "id": identifier,
        "title": title,
        "code_file": "train.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": gpu,
        "dataset_sources": [dataset_source]
    }
    (folder/"kernel-metadata.json").write_text(json.dumps(meta, indent=2))

def poll_status(api, full_slug):
    while True:
        s = api.kernel_status(full_slug)["status"]
        yield s
        if s.lower() in {"complete", "error", "failed"}:
            break
        time.sleep(10)

# ---------- UI ---------------------------------------------------------------
st.set_page_config(page_title="Kaggle Trainer", layout="wide")
st.sidebar.title("Kaggle Trainer")

user = st.sidebar.text_input("Kaggle username", os.getenv("KAGGLE_USERNAME", ""))
dataset_slug = st.sidebar.text_input("Dataset slug", "my-pytorch-model")
kernel_slug = st.sidebar.text_input("Kernel slug", "train-my-model")
epochs = st.sidebar.number_input("Epochs", 1, 100, 10)
gpu = st.sidebar.checkbox("Enable GPU", True)

DS_FOLDER = Path("data_model_dataset")
KERNEL_DIR = Path("kernel")

# Ensure folders exist
DS_FOLDER.mkdir(exist_ok=True)
KERNEL_DIR.mkdir(exist_ok=True)

# Main area
st.title("One-Click Kaggle Training")
log_area = st.empty()
status_area = st.empty()
files_area = st.container()

try:
    api = get_kaggle_api()
    
    # 1. Create / version dataset --------------------------------------------------
    if st.sidebar.button("Create / Update Dataset"):
        identifier = f"{user}/{dataset_slug}"

        # write metadata & config.json
        write_dataset_metadata(DS_FOLDER, dataset_slug, identifier)
        (DS_FOLDER/"config.json").write_text(json.dumps({"epochs": epochs}))

        with log_area:
            try:
                st.write("Creating new dataset…")
                api.dataset_create_new(str(DS_FOLDER), dir_mode="zip")
                st.success("Dataset created successfully!")
            except Exception as e:
                st.write("Dataset exists -- creating new version…")
                api.dataset_create_version(str(DS_FOLDER),
                                       version_notes=f"update at {time.ctime()}",
                                       convert_to_csv=False)
                st.success("Dataset versioned successfully!")

    # 2. Push & run kernel ---------------------------------------------------------
    if st.sidebar.button("Push & Run Kernel"):
        ds_id = f"{user}/{dataset_slug}"
        kn_id = f"{user}/{kernel_slug}"

        write_kernel_metadata(KERNEL_DIR, kernel_slug, kn_id, ds_id, gpu)

        # zip kernel dir (required by API)
        zip_path = KERNEL_DIR.with_suffix(".zip")
        zip_dir(KERNEL_DIR, zip_path)

        with log_area:
            st.write("Pushing kernel…")
            out = api.kernel_push(str(zip_path))
            st.json(out)

            st.write("Polling status ⏳")
            for status in poll_status(api, kn_id):
                status_area.info(f"Current status: {status}")
                time.sleep(2)  # Avoid too frequent updates

            st.success(f"Finished with status: {status}")

    # 3. Refresh status ------------------------------------------------------------
    if st.sidebar.button("Refresh Status"):
        kn_id = f"{user}/{kernel_slug}"
        with log_area:
            info = api.kernel_status(kn_id)
            st.write("Current kernel status:")
            st.json(info)
            status_area.info(f"Status: {info.get('status', 'Unknown')}")

    # 4. Download outputs ----------------------------------------------------------
    if st.sidebar.button("Download Outputs"):
        kn_id = f"{user}/{kernel_slug}"
        out_dir = Path("kernel_outputs")
        out_dir.mkdir(exist_ok=True)
        
        with log_area:
            st.write("Downloading outputs…")
            api.kernel_output(kn_id, str(out_dir))
            st.success(f"Files saved to {out_dir}")
            
            with files_area:
                st.subheader("Available Output Files")
                cols = st.columns(3)
                for i, f in enumerate(out_dir.iterdir()):
                    with cols[i % 3]:
                        st.download_button(
                            f"📥 Download {f.name}", 
                            f.read_bytes(), 
                            file_name=f.name,
                            key=f"download_{i}"
                        )
                        st.write(f"Size: {f.stat().st_size/1024:.1f} KB")

except Exception as e:
    st.error(f"Error: {str(e)}")
    st.info("Make sure your Kaggle credentials are set in the .env file or environment variables.")
