import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(
    current_dir,
    "images",
    "image.jpg"
)
img = cv2.imread(image_path)

if img is None:
    print("Error: Image not found!")
    print(image_path)
    exit()

height, width = img.shape[:2]

max_width = 800
max_height = 600

scale = min(max_width/width, max_height/height)

if scale < 1:
    img = cv2.resize(
        img,
        (int(width*scale), int(height*scale))
    )

cv2.namedWindow("Image", cv2.WINDOW_NORMAL)


cv2.imshow("Original Image", img)
cv2.waitKey(0)


print("Choose operation:")
print("1: Grayscale")
print("2: Flip")
print("3: Brightness")
print("4: Blur")
print("5: Edge Detection")
print("6: Contrast")
print("7: Rotate")
print("8: Sharpen")
print("9: Cartoon")
print("10: Pencil Sketch")
print("11: Histogram Analysis")


choice = int(input("Enter your choice: "))


if choice == 1:
    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    cv2.imshow(
        "Grayscale",
        gray
    )

    output_path = os.path.join(
        current_dir,
        "outputs",
        "grayscale.jpg"
    )

    cv2.imwrite(
        output_path,
        gray
    )

elif choice == 2:
    flipped = cv2.flip(
        img,
        1
    )

    cv2.imshow(
        "Flipped",
        flipped
    )

    output_path = os.path.join(
        current_dir,
        "outputs",
        "flipped.jpg"
    )

    cv2.imwrite(
        output_path,
        flipped
    )

elif choice == 3:
    bright = np.clip(
        img + 50,
        0,
        255
    )

    cv2.imshow(
        "Bright Image",
        bright
    )
    output_path = os.path.join(
        current_dir,
        "outputs",
        "bright.jpg"
    )

    cv2.imwrite(
        output_path,
        bright
    )

elif choice == 4:
    blur = cv2.GaussianBlur(
        img,
        (15, 15),
        0
    )
    cv2.imshow(
        "Blur Image",
        blur
    )

    output_path = os.path.join(
        current_dir,
        "outputs",
        "blur.jpg"
    )

    cv2.imwrite(
        output_path,
        blur
    )

elif choice == 5:
    edges = cv2.Canny(
        img,
        100,
        200
    )
    cv2.imshow(
        "Edge Detection",
        edges
    )

    output_path = os.path.join(
        current_dir,
        "outputs",
        "edges.jpg"
    )

    cv2.imwrite(
        output_path,
        edges
    )


elif choice == 6:
    contrast = cv2.convertScaleAbs(
        img,
        alpha=1.8,
        beta=0
    )

    cv2.imshow("Contrast", contrast)

    output_path = os.path.join(
        current_dir,
        "outputs",
        "contrast.jpg"
    )

    cv2.imwrite(output_path, contrast)


elif choice == 7:

    h, w = img.shape[:2]

    matrix = cv2.getRotationMatrix2D(
        (w//2, h//2),
        45,
        1
    )

    rotated = cv2.warpAffine(
        img,
        matrix,
        (w, h)
    )

    cv2.imshow(
        "Rotated",
        rotated
    )

    output_path = os.path.join(
        current_dir,
        "outputs",
        "rotated.jpg"
    )

    cv2.imwrite(output_path, rotated)

elif choice == 8:

    kernel = np.array([
        [-1,-1,-1],
        [-1, 9,-1],
        [-1,-1,-1]
    ])

    sharp = cv2.filter2D(
        img,
        -1,
        kernel
    )

    cv2.imshow(
        "Sharpen",
        sharp
    )

    output_path = os.path.join(
        current_dir,
        "outputs",
        "sharpen.jpg"
    )

    cv2.imwrite(output_path, sharp)

elif choice == 9:

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.medianBlur(
        gray,
        5
    )

    edges = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        9,
        9
    )

    color = cv2.bilateralFilter(
        img,
        9,
        250,
        250
    )

    cartoon = cv2.bitwise_and(
        color,
        color,
        mask=edges
    )

    cv2.imshow(
        "Cartoon Effect",
        cartoon
    )

    output_path = os.path.join(
        current_dir,
        "outputs",
        "cartoon.jpg"
    )

    cv2.imwrite(
        output_path,
        cartoon
    )

elif choice == 10:

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    inverted = 255 - gray

    blurred = cv2.GaussianBlur(
        inverted,
        (21, 21),
        0
    )

    sketch = cv2.divide(
        gray,
        255 - blurred,
        scale=256
    )

    cv2.imshow(
        "Pencil Sketch",
        sketch
    )

    output_path = os.path.join(
        current_dir,
        "outputs",
        "sketch.jpg"
    )

    cv2.imwrite(
        output_path,
        sketch
    )

elif choice == 11:

    plt.hist(
        img.ravel(),
        256,
        [0, 256]
    )

    plt.title("Image Histogram")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")

    output_path = os.path.join(
        current_dir,
        "outputs",
        "histogram.jpg"
    )

    plt.savefig(output_path)

    plt.show()

    plt.close()

else:
    print("Invalid choice")

cv2.waitKey(0)
cv2.destroyAllWindows()