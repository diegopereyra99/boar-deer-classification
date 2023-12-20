import os
import logging
import requests
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

os.makedirs("logs/", exist_ok=True)
logging.basicConfig(filename="logs/download_logs.txt", level=logging.INFO)      
        
os.makedirs(f"data/imgs", exist_ok=True)
for fn in os.listdir("data/sources"):    
    with open(os.path.join("data/sources", fn), "r") as file:
        clss = fn.split('_')[0]
        logging.info(clss)
        for i, url in tqdm(enumerate(file.readlines())):
            if os.path.exists(f"data/imgs/{clss}_{i}.jpeg"):
                continue
            try:
                # urllib.request.urlretrieve(url.strip(), f"data/{clss}/{i}.jpg")
                r = requests.get(url.strip(), headers={'User-Agent': 'My User Agent 1.0'})
                ext = r.headers["Content-Type"].split("/")[-1].split(";")[0]
                if ext not in ["jpeg", "jpg", "png", "webp"]:
                    logging.error(f"{i}: {ext} extension  /// {url.strip()}")
                    continue
                
                filepath = f"data/imgs/{clss}_{i}.{ext}"
                with open(filepath, 'wb') as outfile:
                    outfile.write(r.content)
                    
                try:
                    im = Image.open(filepath)
                    rgb_im = im.convert('RGB')
                except UnidentifiedImageError as e:
                    logging.error(f"{i}: {e}   ///  {url.strip()}")
                    os.remove(filepath)
                    logging.warning(f"{i}: deleted")
                    raise e
                
                logging.info(f"{i}: OK")
                
            except Exception as e:
                logging.error(f"{i}: {e}   ///  {url.strip()}")
                continue
            
