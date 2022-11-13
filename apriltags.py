import apriltag
import numpy as np
import cv2, math
import argparse


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
    help="path to input image containing AprilTag")
args = vars(ap.parse_args())

print("[INFO] loading image...")
frame = cv2.imread(args["image"])
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# define the AprilTags detector options and then detect the AprilTags
# in the input image
print("[INFO] detecting AprilTags...")
options = apriltag.DetectorOptions(families="tag36h11")
detector = apriltag.Detector(options)
results = detector.detect(gray)
print("[INFO] {} total AprilTags detected".format(len(results)))
# loop over the AprilTag detection results
tags = {}
for r in results:
    # extract the bounding box (x, y)-coordinates for the AprilTag
    # and convert each of the (x, y)-coordinate pairs to integers
    (ptA, ptB, ptC, ptD) = r.corners
    ptB = (int(ptB[0]), int(ptB[1]))
    ptC = (int(ptC[0]), int(ptC[1]))
    ptD = (int(ptD[0]), int(ptD[1]))
    ptA = (int(ptA[0]), int(ptA[1]))
    # draw the bounding box of the AprilTag detection
    cv2.line(frame, ptA, ptB, (0, 255, 0), 2)
    cv2.line(frame, ptB, ptC, (0, 255, 0), 2)
    cv2.line(frame, ptC, ptD, (0, 255, 0), 2)
    cv2.line(frame, ptD, ptA, (0, 255, 0), 2)
    # draw the center (x, y)-coordinates of the AprilTag
    (cX, cY) = (int(r.center[0]), int(r.center[1]))
    cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)
    # draw the tag family on the frame
    tagFamily = "TAG_FAMILY: " + r.tag_family.decode("utf-8")
    tag_id = "TAG-ID: " + str(r.tag_id)
    cv2.putText(frame, tagFamily, (ptA[0], ptA[1] - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(frame, tag_id, (ptA[0], ptA[1] + 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    print(f"[INFO] {tagFamily}, {tag_id}")
    tags[r.tag_id] = (cX, cY)
# show the output frame after AprilTag detection
# print(list(tags.keys()))
# print(tags)
if 0 in tags.keys() and 1 in tags.keys():
    x1, y1 = tags[0]
    x2, y2 = tags[1]
    midY = int((y1 + y2)/2)
    midX = int((x1 + x2)/2)
    try:
        rad_angle = math.atan((y1-y2)/(x2-x1))
    except ZeroDivisionError:
        rad_angle = 1.570796 # force 90deg
    length=.3 * ( np.linalg.norm((x1-x2,y1-y2)))
    
    x3 = int(midX + length * math.sin(rad_angle + math.pi))
    y3 = int(midY + length * math.cos(rad_angle + math.pi))

    print(midX, midY, x3, y3, rad_angle, "("+str(rad_angle*180/math.pi)+")")
    cv2.circle(frame, (midX, midY), 5, (255, 0, 0), -1)
    cv2.circle(frame, (x3, y3), 5, (255, 255, 0), -1)
    cv2.line(frame, (x1,y1), (x2,y2), (0, 0, 255), 2)
    cv2.line(frame, (midX, midY), (x3, y3), (255, 255, 0), 2)

cv2.imshow("Image", frame)
cv2.waitKey(0)