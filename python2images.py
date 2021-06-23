import cv2
#import nunpy as np

original = cv2.imread("3545122068_5bf27ae7c4_b.jpg")
image_to_compare = cv2.imread("3545122068_5bf27ae7c4_c.jpg")

# 1) check if 2 images are equals
if original.shape == image_to_compare.shape:
	print("Image same size and channel")
	difference = cv2.subtract(original, image_to_compare) # tru pixel 2 hinh, va ve su khac nhau
#	cv2.imshow("difference", difference)
	b, g, r = cv2.split(difference)
#	cv2.imshow("b", b) #show hinh chi voi mau blue
#	cv2.imshow("g", g) #show hinh chi voi mau green
#	cv2.imshow("r", r) #show hinh chi voi mau red

	if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r)  == 0:
		print("The image are completetly equal")
	else:
		print("The image are NOT equal")
# 2) 
sift = cv2.xfeatures2d.SIFT_create()
Kp_1,Desc_1 = sift.detectAndCompute(original, None)
Kp_2,Desc_2 = sift.detectAndCompute(image_to_compare, None)

index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(Desc_1, Desc_2, k=2)

print(len(matches))





#cv2.imshow("Orginal", original)
cv2.imshow("duplicate", image_to_compare)
cv2.waitKey(0)
cv2.destroyAllWindows()


