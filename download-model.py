import os
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup

CHUNK_SIZE = 1024 * 1024  # 1 MB

def scrape_files_from_model(model_name):
    # Ensure the model name has no trailing slash
    model_name = model_name.rstrip("/")

    # Construct the URL of the model's page
    model_url = f"https://huggingface.co/{model_name}/tree/main"

    # Send a GET request to the model's page and parse the HTML content
    response = requests.get(model_url)
    soup = BeautifulSoup(response.content, "html.parser")

    # Extract all the file links
    file_links = []
    for link in soup.find_all("a"):
        href = link.get("href")
        if href and "/resolve/main/" in href:
            file_links.append("https://huggingface.co"+href)

    # Return the list of file links
    return file_links


def download_model_files(hub_model_name, destination_folder):
    # Create the destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)

    # Get all the files associated with the model
    files = scrape_files_from_model(hub_model_name)

    # Download each file
    for file_url in files:
        split = file_url.split('/')
        file_name = split[split.__len__()-1]
        print(file_name)
        file_path = os.path.join(destination_folder, file_name)

        # Check if the file already exists
        if os.path.exists(file_path):
            print(f"File {file_name} already exists.")
            continue

        # Check if there was a previous interrupted download
        if os.path.exists(file_path + ".part"):
            print(f"Resuming previous download of {file_name}...")
            headers = {'Range': f"bytes={os.path.getsize(file_path + '.part')}-"}
            mode = 'ab'
        else:
            headers = None
            mode = 'wb'

        # Construct the URL for downloading the file

        # Start the download
        with requests.get(file_url, stream=True, headers=headers) as response:
            response.raise_for_status()

            # Total size of the file
            total_size = int(response.headers.get('content-length', 0))

            # Show the progress bar
            with tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024, desc=file_name, ncols=80) as pbar:
                # Open the file for writing in binary mode
                with open(file_path + ".part", mode) as f:
                    # Write the downloaded data to the file in chunks
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            # Rename the downloaded file without the '.part' extension if download completed
            if os.path.getsize(file_path + ".part") == total_size:
                os.rename(file_path + ".part", file_path)
                print(f"Download of {file_name} completed successfully.")
            else:
                print(f"Download of {file_name} incomplete. Try running the script again to resume.")


# Usage
print("This script saves your download progress in '.part' chunks, so you can safely close and resume any time you want.")
hub_model_name = input("model name (ex. internlm/internlm-7b): ")
model_name = hub_model_name.split('/')[0]
destination_folder = "./"+model_name
download_model_files(hub_model_name, destination_folder)
