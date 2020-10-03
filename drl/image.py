import matplotlib.pyplot as plt
import cv2


def preprocess_image(image):
    image = image[32:193, 8:152]
    image = cv2.resize(image, (84, 84))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image


def plot_image(image):
    plt.imshow(image)
    plt.show()
