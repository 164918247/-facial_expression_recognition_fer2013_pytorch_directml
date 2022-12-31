# 原始数据是csv格式，先转换成jpg格式

import numpy as np
import pandas as pd
import os
from PIL import Image

# 测试集
df = pd.read_csv('./data/fer2013.csv')

for i, row in df.iterrows():
    dir = './image/'
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)

    pixels = row['pixels'].split(' ')
    image_data = np.asarray(pixels, dtype=np.uint8).reshape(48,48)
    img = Image.fromarray(image_data)
    img.save('{}{}_{}_{:0>5d}.jpg'.format(dir, row['Usage'], row['emotion'], i), 'JPEG')
