from serpapi import GoogleSearch
import requests
import os
from PIL import Image
import os
def validate_images(image_directory):
    for root, dirs, files in os.walk(image_directory):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')): 
                file_path = os.path.join(root, file)
                try:
                    with Image.open(file_path) as img:
                        img.verify()  
                except (IOError, SyntaxError) as e:
                    print('Bad file:', file_path)
                    os.remove(file_path)




def download_images(query, num_images, output_dir):
    params = {
        "q": query,
        "tbm": "isch",
        "num": num_images,
        "api_key": "7ee5dfa0faec454c63a7062099bf4910f4befd40d801714ba5f1ed369f800df9"
    }
    non_downloaded_images = []
    search = GoogleSearch(params)
    results = search.get_dict()

    images = results['images_results']

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, img in enumerate(images):
        img_url = img['original']
        try:
            img_data = requests.get(img_url,timeout=15).content
            img_name = os.path.join(output_dir, f"{query}_{i}.jpg")

            with open(img_name, 'wb') as handler:
                handler.write(img_data)
            print(f'Downloaded {i+1}/{len(num_images)}')
        except Exception as e:
            non_downloaded_images.append(img_url)
            print(f'Error downloading ==> {img_url}')
            print('--'*50)
            print(f'Error ===> {e}')
    validate_images(output_dir)
# Usage:
data_path=r'"D:\Projects\Python Devolpment\Data field\python\data scintict\medicals-models\data\my_data\scalp infection scrapping"'
download_images("scalp infection", 100,
data_path)
validate_images(data_path)
