import numpy as np 
import cv2
import urllib
import urllib.request
from requests.utils import quote
from misc_utils import get_file_name

MAX_ATTEMPTS = 2

def image_from_url(url):
    basename = get_file_name(url)
    protocol, website = url.split('//')
    website = quote(website)
    url = f'{protocol}//{website}'
    
    for _ in range(MAX_ATTEMPTS):
        try:
            resp = urllib.request.urlopen(url, timeout=3)
            image = np.asarray(bytearray(resp.read()), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            
            if image is not None:
                return image
        except Exception:
            print(f'W: image "{url}" request failed. Retrying...')

    return None
