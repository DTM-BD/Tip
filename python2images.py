import cv2
#import nunpy as np

original = cv2.imread("181002113456-01-golden-gate-bridge-restricted.jpg")
duplicate = cv2.imread("181002113456-01-golden-gate-bridge-restricted.jpg")

# 1) check if 2 images are equals
if original.shape == duplicate.shape:
	print("Image same size and channel")
	difference = cv2.subtract(original, duplicate)
	b, g, r = cv2.split(difference)

	if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r)  == 0:
		print("The image are completetly equal")

# 2) 



cv2.imshow("Orginal", original)
cv2.imshow("duplicate", duplicate)
cv2.waitKey(0)
cv2.destroyAllWindows()


