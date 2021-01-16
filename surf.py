import cv2
import numpy as np
import time


class SurfTracker:
    def __init__(self, hessianThreshold=300, match=1):
        self.hessianThreshold = hessianThreshold
        self.frame = None
        self.initBB = None
        self.baseImage = None
        self.height = None
        self.width = None
        self.match = match
        self.corners = None


    def init(self, frame, initBB):
        self.frame = frame
        self.initBB = initBB
        (x, y, w, h) = initBB
        self.baseImage = frame[y:y+h, x:x+w]
        self.height = h
        self.width = w
        self.corners = np.array([
            [0, 0],
            [0, self.height - 1],
            [self.width - 1, self.height - 1],
            [self.width - 1, 0]
        ])

    def findHomography(self, im1, im2):
        # Convert images to grayscale
        im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

        # Detect SURF features and compute descriptors.
        surf = cv2.xfeatures2d.SURF_create(self.hessianThreshold, extended=True)

        (kp1, desc1) = surf.detectAndCompute(im1Gray, None)
        (kp2, desc2) = surf.detectAndCompute(im2Gray, None)

        # create a BFMatcher object which will match up the SURF features
        if self.match:
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            matches = bf.match(desc1, desc2, None)

            # Sort the matches in the order of their distance.
            matches = sorted(matches, key=lambda x: x.distance)

            # draw the top N matches
            numGoodMatches = int(len(matches) * 0.15)
            matches = matches[:numGoodMatches]

            imMatches = cv2.drawMatches(im1, kp1, im2, kp2, matches, None)
            # cv2.imwrite("matches1.jpg", imMatches)
            # convert the key points to numpy arrays
            points1 = np.zeros((len(matches), 2), dtype=np.float32)
            points2 = np.zeros((len(matches), 2), dtype=np.float32)

            for i, match in enumerate(matches):
                points1[i, :] = kp1[match.queryIdx].pt
                points2[i, :] = kp2[match.trainIdx].pt
        else:
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(desc1, desc2, k=2)
            # # cv.drawMatchesKnn expects list of lists as matches.
            # img3 = cv2.drawMatchesKnn(im1, kp1, im2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            # plt.imshow(img3), plt.show()
            mkp1 = []
            mkp2 = []
            for m in matches:
                if len(m) == 2 and m[0].distance < m[1].distance * 0.7:
                    m = m[0]
                    mkp1.append(kp1[m.queryIdx])
                    mkp2.append(kp2[m.trainIdx])
            if not mkp1 and not mkp2:
                return None, False
            kp_pairs = zip(mkp1, mkp2)

            mkp1, mkp2 = zip(*kp_pairs)

            points1 = np.float32([kp.pt for kp in mkp1])
            points2 = np.float32([kp.pt for kp in mkp2])

        h1, mask = cv2.findHomography(points1, points2, cv2.RANSAC, 1.5)
        n = 0
        for i in mask:
            if i == 1:
                n = n + 1

        err = True
        if n > 3:
            err = False

        return h1, err

    def update(self, frame):
        h, err = self.findHomography(self.baseImage, frame)
        if err:
            return False, None
        else:
            corners = cv2.perspectiveTransform(np.float32([self.corners]), h)[0]
            # Find the bounding rectangle
            box = cv2.boundingRect(corners)
            return True, box


# main
start = time.time()
img1_color = cv2.imread(r"D:\Project\surf\im1.jpg", cv2.IMREAD_COLOR)  # Image to be aligned.
img2_color = cv2.imread(r"D:\Project\surf\im2.jpg", cv2.IMREAD_COLOR)  # Reference image.
