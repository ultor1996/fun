import cv2 as cv
r='C:/Users/shiva/OneDrive/Desktop/fun/dips.mp4'
cap = cv.VideoCapture(r)
fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter('resized_video.mp4', fourcc, 30.0, (400,600))
while cap.isOpened():
    ret,frame=cap.read()
    resize=cv.resize(frame,(400, 600),interpolation=cv.INTER_AREA)
    out.write(resize)
    cv.imshow('resized',resize)
    if cv.waitKey(0) & 0xFF == ord('q'):
        break
cap.release()
out.release()
cv.destroyAllWindows()
