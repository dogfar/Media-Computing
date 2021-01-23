import numpy as np
import scipy.ndimage as nd
from imageio import imread, imwrite
import matplotlib.pyplot as plt
from numba import jit
import cv2

np.set_printoptions(threshold = np.inf) 

def rotat(img, back = False):
	if back == False:
		return np.rot90(img, 1, (0, 1))
	else:
		return np.rot90(img, 3, (0, 1))


def pixel_distance(img, i1, j1, i2, j2):
	return np.sum(np.abs(img[i1][j1] - img[i2][j2]))

@jit
def getmin_for_vertical(M, c, i, j):
	if j == 0:
		if M[i-1][j] < M[i-1][j+1]:
			return j
		else:
			return j+1
	elif j == c-1:
		if M[i-1][j-1] < M[i-1][j]:
			return j-1
		else:
			return j
	else:
		if M[i-1][j-1] <= M[i-1][j] and M[i-1][j-1] <= M[i-1][j+1]:
			return j-1
		elif M[i-1][j] <= M[i-1][j-1] and M[i-1][j] <= M[i-1][j+1]:
			return j
		else:
			return j+1

def getmin_for_vertical_forward(M, img, c, i, j):
	if j > 0 and j < c-1:
		left = M[i-1][j-1] + pixel_distance(img, i, j+1, i, j-1) + pixel_distance(img, i-1, j, i, j-1)
		middle = M[i-1][j] + pixel_distance(img, i, j+1, i, j-1)
		right = M[i-1][j+1] + pixel_distance(img, i, j+1, i, j-1) + pixel_distance(img, i-1, j, i, j+1)
		if left <= middle and left <= right:
			return j-1, left
		elif middle <= left and middle <= right:
			return j, middle
		else:
			return j+1, right

	elif j == 0:
		middle = M[i-1][j] + np.sum(np.abs(img[i][j+1]))
		right = M[i-1][j+1] + pixel_distance(img, i-1, j, i, j+1)
		if middle <= right:
			return j, middle
		else:
			return j+1, right

	else:
		left = M[i-1][j-1] + pixel_distance(img, i-1, j, i, j-1)
		middle = M[i-1][j] + np.sum(np.abs(img[i][j-1]))
		if left <= middle:
			return j-1, left
		else:
			return j, middle


def cal_energy(img):
	kernel_x = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
	kernel_y = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
	kernel_x = np.stack([kernel_x] * 3, axis=2)
	kernel_y = np.stack([kernel_y] * 3, axis=2)

	img = img.astype('float32') # to be removed
	conv_res = np.absolute(nd.convolve(img, kernel_x)) + np.absolute(nd.convolve(img, kernel_y))
	e_img = conv_res.sum(axis=2)
	return e_img

@jit
def cal_energy_gradient(img):
	e_img = np.zeros((img.shape[0], img.shape[1]))
	padding_img = np.zeros((img.shape[0] + 2, img.shape[1] + 2, img.shape[2]))
	padding_img[1 : img.shape[0] + 1, 1 : img.shape[1] + 1] = img[:, :]
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			e_img[i][j] = np.sum(np.abs(padding_img[i + 1][j + 1] - padding_img[i + 1][j + 2])) \
						+ np.sum(np.abs(padding_img[i + 1][j + 1] - padding_img[i + 1][j])) \
						+ np.sum(np.abs(padding_img[i + 1][j + 1] - padding_img[i][j + 1])) \
						+ np.sum(np.abs(padding_img[i + 1][j + 1] - padding_img[i + 2][j + 1]))

	return e_img

@jit
def entropy(padding_img, i, j):
	entropy_d = {}
	for p in range(i - 4, i + 5):
		for q in range(j - 4, j + 5):
			if padding_img[p][q] not in entropy_d:
				entropy_d[padding_img[p][q]] = 1
			else:
				entropy_d[padding_img[p][q]] += 1
	return entropy_d

@jit
def calc_ans(entropy_d):
	e = 0.0
	for k in entropy_d:
		e += (-1.0 * k * np.log (entropy_d[k] / 81) / np.log(2) )

	return e

def cal_energy_entropy(img):
	gray_img = (cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
	gray_img = gray_img.astype(int)
	e_img = np.zeros(gray_img.shape)
	padding_img = np.zeros((img.shape[0] + 8, img.shape[1] + 8))
	padding_img[4 : img.shape[0] + 4, 4 : img.shape[1] + 4] = gray_img[:, :]
	return step_for_jit(e_img, img.shape[0], img.shape[1], padding_img)

@jit
def step_for_jit(e_img, r, c, padding_img):
	for i in range(4, r + 4):
		for j in range(4, c + 4):
			entropy_d = entropy(padding_img, i, j)
			e_img[i - 4][j - 4] = calc_ans(entropy_d)

	return e_img


def addfaces(e_img, faces):
	m = e_img.max()
	for (x, y, w, h) in faces:
		for p in range(y, y+h):
			for q in range(x, x+w):
				e_img[p][q] += 2 * m
	return e_img

def add_mask(e_img, mask_protect, mask_remove):
	m = e_img.max()
	for i in range(mask_remove.shape[0]):
		for j in range(mask_remove.shape[1]):
			e_img[i][j] -= mask_remove[i][j] * e_img.shape[0] * e_img.shape[1] * m
			e_img[i][j] += e_img.shape[0] * e_img.shape[1] * mask_protect[i][j] * m
	return e_img

def min_seam_vertical(e_img):
	M = e_img.copy()
	path = np.zeros(e_img.shape, dtype=np.int)
	
	r, c = e_img.shape
	for i in range(1, r):
		for j in range(0, c):
			path[i][j] = getmin_for_vertical(M, c, i, j)
			M[i][j] += M[i-1][path[i][j]]

	idx = M[r-1].argmin()
	c_list = [idx]
	for j in range(r-1, 0, -1):
		c_list.append(path[j][idx])
		idx = path[j][idx]

	c_list.reverse()
	return c_list

def min_seam_vertical_for_opt(e_img):
	M = e_img.copy()
	path = np.zeros(e_img.shape, dtype=np.int)
	
	r, c = e_img.shape
	for i in range(1, r):
		for j in range(0, c):
			path[i][j] = getmin_for_vertical(M, c, i, j)
			M[i][j] += M[i-1][path[i][j]]

	idx = M[r-1].argmin()
	min_energy = M[r-1][idx]
	c_list = [idx]
	for j in range(r-1, 0, -1):
		c_list.append(path[j][idx])
		idx = path[j][idx]

	c_list.reverse()
	return c_list, min_energy

def min_seam_vertical_forward(img):
	M = np.zeros((img.shape[0], img.shape[1]))
	path = np.zeros((img.shape[0], img.shape[1]), dtype=np.int)
	
	r, c = M.shape
	for i in range(1, r):
		for j in range(0, c):
			path[i][j], en = getmin_for_vertical_forward(M, img, c, i, j)
			M[i][j] += en

	idx = M[r-1].argmin()
	c_list = [idx]
	for j in range(r-1, 0, -1):
		c_list.append(path[j][idx])
		idx = path[j][idx]

	c_list.reverse()
	return c_list

def seam_carving_vertical(img, c_list):
	r, c, ch = img.shape
	mask = np.ones((r, c), dtype = bool)

	for i in range(0, r):
		mask[i][c_list[i]] = False 

	mask = np.stack([mask] * ch, axis=2)
	img = img[mask].reshape((r, c - 1, ch))
	return img

def seam_carve_mask(img, c_list, mp, mr):
	r, c, ch = img.shape
	mask = np.ones((r, c), dtype = bool)

	for i in range(0, r):
		mask[i][c_list[i]] = False 

	mp = mp[mask].reshape((r, c - 1))
	mr = mr[mask].reshape((r, c - 1))
	mask = np.stack([mask] * ch, axis=2)
	img = img[mask].reshape((r, c - 1, ch))
	return img, mp, mr

def seam_extending_vertical(img, seam_list): # should get extra 1 of 1.5
	r, c, ch = img.shape
	new_img = np.zeros([r, c+len(seam_list), ch], dtype = np.uint8)

	s_list_r = np.array(seam_list).T
	for i in range(len(s_list_r)): # row number
		for j in range(0, len(seam_list)): # seam number
			for p in range(0, j):
				if s_list_r[i][p] <= s_list_r[i][j]:
					s_list_r[i][j] += 1

	for i in range(0, r):
		s_list_r[i].sort()
		pos = s_list_r[i].copy() 
		for j in range(len(pos)):
			pos[j] += j

		nowp = 0
		for p in range(c + len(seam_list)):
			if nowp < len(pos):
				if p < pos[nowp]:
					new_img[i][p] = img[i][p-nowp]
				elif p == pos[nowp]:
					new_img[i][p] = (img[i][p-nowp-1] + img[i][p-nowp]) / 2
					nowp += 1
			else:
				new_img[i][p] = img[i][p-nowp]

	return new_img


def carve(img, outfile, nowr, nowc, entropy = False, vertical = True):
	if vertical == False:
		cutc = img.shape[0] - nowr
		img = rotat(img)
	else:
		cutc = img.shape[1] - nowc

	print(cutc)
	for i in range(cutc):
		print(i)
		if entropy == False:
			new_img = seam_carving_vertical(img, min_seam_vertical(cal_energy(img)))
		else:
			new_img = seam_carving_vertical(img, min_seam_vertical(cal_energy_entropy(img)))
		img = new_img

	if vertical == False:
		img = rotat(img, back = True)
	imwrite(outfile, img)

def optimal_carve(img, outfile, nowr, nowc):
	r_cut = img.shape[0] - nowr
	c_cut = img.shape[1] - nowc
	T = np.zeros((r_cut + 1, c_cut + 1))
	img_history = []
	img_history.append(img)

	print(r_cut, c_cut)

	for i in range(0, r_cut + 1):
		for j in range(0, c_cut + 1):
			print(i, j)
			if i == 0 and j == 0:
				continue
			if i == 0:
				img_tmp = img_history[j-1].copy()

				e_img = cal_energy_gradient(img_tmp)
				c_list, min_energy = min_seam_vertical_for_opt(e_img)
				new_img = seam_carving_vertical(img_tmp, c_list)

				img_history.append(new_img)
				T[i][j] = T[i][j - 1] + min_energy

			elif j == 0:
				img_tmp = rotat(img_history[(i - 1) * (c_cut + 1)].copy())

				e_img = cal_energy_gradient(img_tmp)
				c_list, min_energy = min_seam_vertical_for_opt(e_img)
				new_img = seam_carving_vertical(img_tmp, c_list)

				new_img = rotat(new_img, back = True)

				img_history.append(new_img)
				T[i][j] = T[i - 1][j] + min_energy
			else:
				# vertical first
				img_tmp_left = img_history[i  * (c_cut + 1) + j - 1].copy()
				e_img_left = cal_energy_gradient(img_tmp_left)
				c_list_left, min_energy_left = min_seam_vertical_for_opt(e_img_left)

				img_tmp_up = rotat(img_history[(i - 1) * (c_cut + 1) + j].copy())
				e_img_up = cal_energy_gradient(img_tmp_up)
				c_list_up, min_energy_up = min_seam_vertical_for_opt(e_img_up)

				left_energy = min_energy_left + T[i][j - 1]
				up_energy = min_energy_up + T[i - 1][j]

				if left_energy < up_energy:
					new_img = seam_carving_vertical(img_tmp_left, c_list_left)
					img_history.append(new_img)
					T[i][j] = T[i][j - 1] + left_energy

				else:
					new_img = seam_carving_vertical(img_tmp_up, c_list_up)
					new_img = rotat(new_img, back = True)
					img_history.append(new_img)
					T[i][j] = T[i - 1][j] + up_energy

	imwrite(outfile, img_history[-1])

def extend(img, outfile, nowr, nowc, vertical = True):
	img_copy = img.copy()
	if vertical == False:
		extc = nowr - img.shape[0]
		img = rotat(img)
	else:
		extc = nowc - img.shape[1]

	seam_list = []
	for i in range(extc):
		one_seam = min_seam_vertical(cal_energy(img_copy))
		seam_list.append(one_seam)
		new_img = seam_carving_vertical(img_copy, one_seam)
		img_copy = new_img

	ext_img = seam_extending_vertical(img, seam_list)
	imwrite(outfile, ext_img)


def face_detect_cut(img_carve, outfile, nowr, nowc, vertical = True):
	face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	if vertical == False:
		cutc = img_carve.shape[0] - nowr
		img_carve = rotat(img_carve)
	else:
		cutc = img_carve.shape[1] - nowc

	print(cutc)
	for i in range(cutc):
		print(i)
		gray = cv2.cvtColor(img_carve, cv2.COLOR_RGB2GRAY)
		faces = face_cascade.detectMultiScale(gray, 1.1, 4)
		e_img = cal_energy(img_carve)
		e_img = addfaces(e_img, faces)
		new_img = seam_carving_vertical(img_carve, min_seam_vertical(e_img))
		img_carve = new_img

	if vertical == False:
		img = rotat(img, back = True)
	imwrite(outfile, img_carve)
'''
#img1 = imread('pic1.jpg').astype('float32')
#img2 = imread('pic2.jpg').astype('float32')
#carve(img1, 'another_carved_entropy.jpg', img1.shape[0], int(img1.shape[1] * 0.6), entropy = True, vertical = True)
#carve(img2, 'pic2_carved_entropy.jpg', int(0.6 * img2.shape[0]), img2.shape[1], entropy = True, vertical = False)
'''

img_bad = imread('jimmy.png').astype('uint8')
face_detect_cut(img_bad, 'bad_jimmy.png', img_bad.shape[0], int(0.6 * img_bad.shape[1]), vertical = True)

'''
#img2 = imread('pic1.jpg').astype('float32')
#extend(img2, 'extended.jpg', img2.shape[0], int(img2.shape[1] * 1.25), vertical = True)
'''

'''
#img3 = imread('girl.jpg').astype('uint8')
#img3_2 = img3.copy().astype('float32')
#carve(img3_2, 'person_carved.jpg', img3_2.shape[0], int(img3_2.shape[1] * 0.6), vertical = True)
#face_detect_cut(img3, 'person_carved_detect.jpg', img3.shape[0], int(img3.shape[1] * 0.6), vertical = True) # input has to be uint8
'''

'''
img4 = imread('Beach.png').astype('float32')
img4_copy = img4.copy()

mask = imread('beach_girl_mask.png').astype('float32') / 255
mask_protect = mask[..., 0]
mask_remove = mask[..., 1]

while np.sum(mask_remove) > 0:
	e_img = cal_energy(img4)
	e_img = add_mask(e_img, mask_protect, mask_remove)
	new_img, mask_protect, mask_remove = seam_carve_mask(img4, min_seam_vertical(e_img), mask_protect, mask_remove)
	img4 = new_img

imwrite('protected.png', img4)
'''

'''
img5 = imread('bench3.png').astype('float32')
for i in range(200):
	print(i)
	new_img = seam_carving_vertical(img5, min_seam_vertical_forward(img5))
	img5 = new_img
imwrite('forward3.png', img5)
'''

'''
img6 = imread('butterfly.png')
img6_copy = img6.copy()
#optimal_carve(img6, 'cut_butterfly.png', 150, 250)
carve(img6_copy, 'tmp.png', 150, img6_copy.shape[1], entropy = False, vertical = False)
tmp = imread('tmp.png')
carve(tmp, 'hard_cut.png', 150, 250, entropy = False, vertical = True)
'''