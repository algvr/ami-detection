from PIL import Image
import random
import numpy as np

with Image.open('images/backgrounds/wall-01.jpg').convert('RGBA') as bg:
    with Image.open('images/ecg-paper/ecg-paper-01.png').convert('RGBA') as fg:
        for i in range(1):
            rot = fg.rotate(np.clip(random.gauss(mu=0, sigma=10), -20, 20), expand=True, fillcolor=(0, 0, 0, 0))
            bg.paste(rot, (500, 500), rot)
            bg.show()
