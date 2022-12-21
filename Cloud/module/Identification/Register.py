
import cv2
import os
import time

person='tyz'

video_path='userdata/video/'+person
origin='userdata/origin/'+person
if not os.path.exists(video_path):
    os.mkdir(video_path)
if not os.path.exists(origin):
    os.mkdir(origin)

video_capture = cv2.VideoCapture(0)
record = 0
counter = 0
timer = 0


r_s = time.time()
ret, frame_t = video_capture.read()
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
FrameSize = (frame_t.shape[1], frame_t.shape[0])
v_s_path = video_path + '/' + person + '.avi'
outfile = cv2.VideoWriter(v_s_path, fourcc, 30., FrameSize)



while (video_capture.isOpened()):
    ret, frame = video_capture.read()
    font = cv2.FONT_HERSHEY_TRIPLEX
    Intro = "adjust, 'a' to start,'ESC' to quit"
    cv2.putText(frame, Intro, (40, 40), font, 0.5, (255, 0, 0), 0, 1)
    cv2.imshow('Capture', frame)

    if cv2.waitKey(50) & 0xFF == ord('a'):
        while True:
            record +=1
            ret, frame = video_capture.read()
            outfile.write(frame)
            cv2.imwrite(origin+'/'+str(record)+'.jpg',frame)
            Intro1 = "shake your head, wait 10 seconds to quit"
            cv2.putText(frame, Intro1, (40,40), font, 0.5, (255, 0, 0), 0, 1)
            cv2.imshow('Capture', frame)
            cv2.waitKey(1)
            if record == 180:
                break

    if cv2.waitKey(40) & 0xFF == 27:
        cv2.destroyAllWindows()
        break

video_capture.release()
outfile.release()
cv2.destroyAllWindows()
print("/------Finished Recording------/")
