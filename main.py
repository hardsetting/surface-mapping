import cv2
import numpy as np

def parsePoints(filename):
	fh = open(filename, "r")
	lines = fh.readlines()
	fh.close()
	
	pairs = [line.strip().split(' ') for line in lines]
	points = [(float(x), float(y)) for x, y in pairs]
	return np.array(points, dtype=np.float32)

class MappedImage:
	def __init__(self, filename, srcPoints, dstPoints, mask=None):
		if isinstance(srcPoints, str):
			srcPoints = parsePoints(srcPoints)
	
		if isinstance(dstPoints, str):
			dstPoints = parsePoints(dstPoints)
	
		self.img = cv2.imread(filename, cv2.IMREAD_COLOR).astype(float) / 255
		self.H, _ = cv2.findHomography(srcPoints, dstPoints)
		self.invH, _ = cv2.findHomography(dstPoints, srcPoints)
		
		if mask is not None:
			self.mask = cv2.imread(mask, cv2.IMREAD_COLOR)
			self.mask = self.mask.astype(float) / 255
		else:
			self.mask = None
	
	def size(self):
		return (self.img.shape[1], self.img.shape[0])
	
	def bird_view(self, size = None, offset = (0, 0)):
		Ht = np.eye(3, 3, dtype=np.float32)
		Ht[0][2] = offset[0]
		Ht[1][2] = offset[1]
	
		H = np.matmul(Ht, self.H)

		res = cv2.warpPerspective(self.img, H, size)
		return res
		
	def apply(self, img):
		H = np.matmul(self.invH, img.H)
		warped = cv2.warpPerspective(img.img, H, self.size()) 
		
		if self.mask is not None:
			print(warped.shape)
			print(self.mask.shape)
			warped = cv2.multiply(1.0 - self.mask, warped)
			
		return np.where(warped > 0, warped, self.img)	

if __name__ == "__main__":
	inputs = [
		("data/photo1.jpg", "data/srcPoints1.txt", "data/photo1_mask.png"),
		("data/photo2.jpg", "data/srcPoints2.txt", "data/photo2_mask.png"),
		("data/photo3.jpg", "data/srcPoints3.txt", "data/photo3_mask.png")
	]

	dstPoints = parsePoints("data/dstPoints.txt")
	imgs = [MappedImage(input[0], input[1], dstPoints, mask=input[2]) for input in inputs]
	map_med = MappedImage("data/map.jpg", "data/srcPointsMap.txt", "data/dstPointsMap.txt")
	map_lrg = MappedImage("data/murales.jpg", "data/srcPointsMurales.txt", "data/dstPointsMurales.txt")

	for img in imgs:
		cv2.imshow("Medium", img.apply(map_med))
		# cv2.imshow("Large", img.apply(map_lrg))
		cv2.waitKey()
	