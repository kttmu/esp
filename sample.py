import cv2

img = cv2.imread("./bdd100k_seg/bdd100k/seg/images/train/0004a4c0-d4dff0ad.jpg")
print(img.shape)
cv2.imshow("./bdd100k_seg/bdd100k/seg/images/train/0004a4c0-d4dff0ad.jpg", img)
cv2.waitKey(0)
