import cv2 
import numpy as np 
import os
image_folder = "./output/pics/"
output_video = "./output/output.mp4"
images = os.listdir(image_folder)
images = [x for x in images if x.endswith(".png")]
def f(x):
    try:
        return int(x.split(".")[0])
    except:
        print("This is not a valid file",x)
        return 0
images = sorted(os.listdir(image_folder), key = f)
image_path = os.path.join(image_folder, images[1])
first_image = cv2.imread(image_path)
height, width, layers = first_image.shape

fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter(output_video, fourcc, 50.0, (width, height))
from tqdm import tqdm
for image_name in tqdm(images[1:]):
    image_path = os.path.join(image_folder, image_name)
    image = cv2.imread(image_path)
    video_writer.write(image)

video_writer.release()
cv2.destroyAllWindows()