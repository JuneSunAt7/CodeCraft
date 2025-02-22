import os
import urllib.request

def download_code_search_net(save_path='code_search_net', language='python'):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    url = f"https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/{language}.zip"
    zip_file_path = os.path.join(save_path, f"{language}.zip")

    try:
        print(f"Downloading dataset from {url}...")
        urllib.request.urlretrieve(url, zip_file_path)
        print(f"Dataset downloaded to {zip_file_path}")
    except Exception as e:
        print(f"Error downloading dataset: {e}")

if __name__ == "__main__":
    download_code_search_net()