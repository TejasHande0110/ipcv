
# Reading Image

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

image = mpimg.imread('/content/car.jpeg')
plt.imshow(image)

image_new = mpimg.imread('/content/sl.jpeg')
plt.imshow(image_new)

image_gray = mpimg.imread('/content/car_gray.png')
plt.imshow(image_gray)

import numpy as np

# Grayscale

def grayscale(image):
  return np.dot(image[...,:3],[0.2989, 0.5870, 0.1140]).astype(np.uint8)

gray_image = grayscale(image)
plt.imshow(gray_image, cmap = "gray")

def greyscale(image):
    og_shape=image.shape
    flat_img=image.flatten()
    L=flat_img.max()
    new_img=[]
    for pixel in flat_img:
        new_img.append(L-1-pixel)
    return np.array(new_img).reshape(og_shape).astype(np.uint8)

greyscale(image)

def grayscale(image):
  plt.imshow(np.dot(image[...,:3],[0.2989,0.5870,0.1140]),cmap="gray")

print(gray_image)

gray_image = grayscale(image_new)


# Merge and Split

def merge_regions(regions):
    while True:
        merged = False
        new_regions = []
        while regions:
            current = regions.pop()
            was_merged = False
            for idx, region in enumerate(new_regions):
                if abs(np.mean(region) - np.mean(current)) < 5:
                    new_regions[idx] = np.vstack([region, current])
                    was_merged = True
                    merged = True
                    break
            if not was_merged:
                new_regions.append(current)
        regions = new_regions
        if not merged:
            break
    return regions


def split_and_merge(image, num_regions):
    rows, cols = image.shape
    step = rows // num_regions
    regions = [np.arange(i, min(i + step, rows)) for i in range(0, rows, step)]
    regions = merge_regions(regions)

    output_image = np.zeros_like(image)
    for region in regions:
        for row in region:
            output_image[row, :] = ((np.mean(image[region, :]) - image[row, :]) < 10) * 255
    return output_image


split_merged_image = split_and_merge(gray_image, 4)

plt.imshow(split_merged_image, cmap="gray")

