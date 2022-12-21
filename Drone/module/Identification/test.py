import cv2
from insightface.app import FaceAnalysis

app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(160, 160))
img=cv2.imread('img/50 Cent/1.jpg')
faces=app.get(img)
kps=faces[0].kps[0].astype('int')
img[kps[1]-10:kps[1]+10,kps[0]-10:kps[0]+10,0]=0
img[kps[1]-10:kps[1]+10,kps[0]-10:kps[0]+10,1]=255
img[kps[1]-10:kps[1]+10,kps[0]-10:kps[0]+10,2]=0

cv2.namedWindow("result",cv2.WINDOW_NORMAL)
cv2.imshow("result",img)
cv2.waitKey(0)

