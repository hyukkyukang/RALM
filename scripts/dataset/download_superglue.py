import os
import shutil
import urllib.request
import zipfile

# URL for the SuperGLUE dataset
dataset_url = "https://dl.fbaipublicfiles.com/glue/superglue/data/v2/combined.zip"
zip_filename = "combined.zip"
extract_folder = "superglue_data"

# Folders to remove
FOLDERS_TO_REMOVE = ["AX-b", "AX-g"]


# Function to download the dataset
def download_dataset(url, filename):
    print(f"Downloading {filename}...")
    urllib.request.urlretrieve(url, filename)
    print("Download complete.")


# Function to extract the dataset
def extract_dataset(zip_path, extract_to):
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction complete.")


# Function to remove the zip file
def cleanup_zipfile(zip_path):
    print(f"Removing {zip_path}...")
    os.remove(zip_path)
    print("Zip file cleanup complete.")


# Function to remove folders
def remove_folders(base_path, folders):
    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        if os.path.exists(folder_path):
            print(f"Removing {folder_path}...")
            shutil.rmtree(folder_path)
        else:
            print(f"{folder_path} does not exist.")
    print("Folder cleanup complete.")


# Main execution
if __name__ == "__main__":
    download_dataset(dataset_url, zip_filename)
    extract_dataset(zip_filename, extract_folder)
    cleanup_zipfile(zip_filename)
    remove_folders(extract_folder, FOLDERS_TO_REMOVE)
    print("SuperGLUE dataset is ready!")
