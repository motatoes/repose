import cv2
import sys

# cascPath = sys.argv[1]
cascPath = "haar/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    # for (x, y, w, h) in faces:
    #     cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    blur = cv2.GaussianBlur(frame,(75,75),0)

    for (x, y, w, h) in faces:

        for row in range(y, y+h):
            for col in range(x, x+w):
                blur[row,col,:] = frame[row, col,:]

    blur = cv2.resize(blur, (frame.shape[1]//2, frame.shape[0]//2) )
    frame = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2) )


    # Display the resulting frame
    cv2.imshow('Video', blur)
    cv2.imshow('Video2', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
