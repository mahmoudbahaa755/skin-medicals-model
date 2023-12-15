from selenium import webdriver
from selenium.webdriver.common.by import By
import os
import time
import wget
import logging
import shutil

def setup_logging():
    logging.basicConfig(filename='image_downloader.log', level=logging.INFO)

def setup_driver():
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--headless")
    chrome_options.add_experimental_option("excludeSwitches", ['enable-automation'])
    return webdriver.Chrome(options=chrome_options)

def scroll_to_bottom(driver):
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

def download_image(img, directory, counter, downloaded_images):
    try:
        src = img.get_attribute("src")
        if src != None and src not in downloaded_images:
            temp_file = wget.download(src)
            if os.path.getsize(temp_file) >= 2000:  # size in bytes
                shutil.move(temp_file, f"{directory}/{directory}_{counter}.png")
                downloaded_images.add(src)
                counter += 1
            else:
                os.remove(temp_file)
    except Exception as e:
        logging.error(f"Error occurred: {e}")
    return counter

def download_images(url, directory):
    driver = setup_driver()

    driver.get(url)

    SCROLL_PAUSE_TIME = 0.5
    last_height = 0

    if not os.path.exists(directory):
        os.makedirs(directory)

    counter = 1
    downloaded_images = set()

    while True:
        scroll_to_bottom(driver)
        time.sleep(SCROLL_PAUSE_TIME)

        imgs = driver.find_elements(By.TAG_NAME, 'img')
        for i in imgs:
            counter = download_image(i, directory, counter, downloaded_images)

        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height or counter > 1000:
            break
        last_height = new_height

    driver.close()

setup_logging()
url='https://www.google.com/search?q=scalp+infection&sca_esv=586961924&tbm=isch&sxsrf=AM9HkKkonZRAEa_JquAnJjy_5yl3qNYmzg:1701434698943&source=lnms&sa=X&ved=2ahUKEwj7ztqWou6CAxW_U6QEHXFvCagQ_AUoAXoECAMQAw'
download_images(url, 'Scalp_infectedGoogle')