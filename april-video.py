import cv2, apriltag, math
import numpy as np
cap = cv2.VideoCapture(-1)

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    options = apriltag.DetectorOptions(families="tag36h11")
    detector = apriltag.Detector(options)
    results = detector.detect(gray)

    tags = {}
    for r in results:
        # extract the bounding box (x, y)-coordinates for the AprilTag
        # and convert each of the (x, y)-coordinate pairs to integers
        (pt1, pt2, pt3, pt4) = r.corners
        pt2 = (int(pt2[0]), int(pt2[1]))
        pt3 = (int(pt3[0]), int(pt3[1]))
        pt4 = (int(pt4[0]), int(pt4[1]))
        pt1 = (int(pt1[0]), int(pt1[1]))
        # draw the bounding box of the AprilTag detection
        cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
        cv2.line(frame, pt2, pt3, (0, 255, 0), 2)
        cv2.line(frame, pt3, pt4, (0, 255, 0), 2)
        cv2.line(frame, pt4, pt1, (0, 255, 0), 2)
        # draw the center (x, y)-coordinates of the AprilTag
        (cx, cy) = (int(r.center[0]), int(r.center[1]))
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        # draw the tag family on the frame
        tag_family = "TAG-FAMILY: " + r.tag_family.decode("utf-8")
        tag_id = "TAG-ID: " + str(r.tag_id)
        cv2.putText(frame, tag_family, (pt1[0], pt1[1] - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, tag_id, (pt1[0], pt1[1] + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        print(f"[INFO] {tag_family}, {tag_id}")
        tags[r.tag_id] = (cx, cy)
    # show the output frame after AprilTag detection
    # print(list(tags.keys()))
    # print(tags)
    if 0 in tags.keys() and 1 in tags.keys():
        x1, y1 = tags[0]
        x2, y2 = tags[1]
        midy = int((y1 + y2)/2)
        midx = int((x1 + x2)/2)
        try:
            angle_rad = math.atan((y1-y2)/(x2-x1))
        except ZeroDivisionError:
            angle_rad = 1.570796 # force 90deg
        length=.3 * ( np.linalg.norm((x1-x2,y1-y2)))
        
        x3 = int(midx + length * math.sin(angle_rad + math.pi))
        y3 = int(midy + length * math.cos(angle_rad + math.pi))

        print(midx, midy, x3, y3, angle_rad, "("+str(angle_rad*180/math.pi)+")")
        cv2.circle(frame, (midx, midy), 5, (255, 0, 0), -1)
        cv2.circle(frame, (x3, y3), 5, (255, 255, 0), -1)
        cv2.line(frame, (x1,y1), (x2,y2), (0, 0, 255), 2)
        cv2.line(frame, (midx, midy), (x3, y3), (255, 255, 0), 2)
    
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

cap.release()
cv2.destroyAllWindows()