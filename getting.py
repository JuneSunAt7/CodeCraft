import requests
import os


def download_code_from_github(save_path='assets', language='python', max_repos=10):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    headers = {'Authorization': 'token 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'}
    url = f"https://api.github.com/search/repositories?q=language:{language}+for+beginners&sort=forks&order=desc"
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(f"Error fetching repositories: {response.status_code}")
        return

    data = response.json()
    repos_downloaded = 0

    for repo in data['items']:
        contents_url = repo['contents_url'].replace('{+path}', '')
        get_repo_contents(contents_url, save_path, headers)
        repos_downloaded += 1
        if repos_downloaded >= max_repos:
            break
# fuck

def get_repo_contents(url, save_path, headers):
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        contents = response.json()
        for item in contents:
            full_save_path = os.path.join(save_path, item['name'])

            if item['type'] == 'file' and item['name'].endswith('.py'):
                file_response = requests.get(item['download_url'])
                if file_response.status_code == 200:
                    os.makedirs(os.path.dirname(full_save_path), exist_ok=True)

                    with open(full_save_path, 'w', encoding='utf-8') as f:
                        f.write(file_response.text)

            elif item['type'] == 'dir':
                # recursed check dir
                get_repo_contents(item['url'], os.path.join(save_path, item['name']), headers)

if __name__ == "__main__":
    download_code_from_github()