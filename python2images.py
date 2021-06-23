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

good_points=[]
for m,n in matches:
	if m.distance < 0.6*n.distance:
		good_points.append(m)
print(len(good_points))

#print(len(matches)) # in ra cac mtach giua 2 hinh
result = cv2.drawMatchesKnn(original, Kp_1, image_to_compare, Kp_2, matches, None)

#cv2.imshow("result", cv2.resize(result, None, fx=0.4, fy=0.4)) # show ra hinh anh, match giua 2 hinh





cv2.imshow("Orginal", cv2.resize(original, None, fx=0.4, fy=0.4))
cv2.imshow("duplicate", cv2.resize(image_to_compare, None, fx=0.4, fy=0.4))
cv2.waitKey(0)
cv2.destroyAllWindows()


