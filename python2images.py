import cv2
#import nunpy as np

original = cv2.imread("3545122068_5bf27ae7c4_b.jpg")
duplicate = cv2.imread("3545122068_5bf27ae7c4_c.jpg")

# 1) check if 2 images are equals
if original.shape == duplicate.shape:
	print("Image same size and channel")
	difference = cv2.subtract(original, duplicate) # tru pixel 2 hinh, va ve su khac nhau
#	cv2.imshow("difference", difference)
	b, g, r = cv2.split(difference)
#	cv2.imshow("b", b) #show hinh chi voi mau blue
#	cv2.imshow("g", g) #show hinh chi voi mau green
#	cv2.imshow("r", r) #show hinh chi voi mau red

	if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r)  == 0:
		print("The image are completetly equal")

# 2) 



#cv2.imshow("Orginal", original)
cv2.imshow("duplicate", duplicate)
cv2.waitKey(0)
cv2.destroyAllWindows()


