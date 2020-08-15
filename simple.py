#  http://layer0.authentise.com/segment-background-using-computer-vision.html

index = 0

while (1):
    index = index + 1

    ret, frame = cap.read()
    if ret == False:
        break;

    frame = cv.flip(frame, 1)

    blur = cv.GaussianBlur(frame, (5, 5), 0)
    # rect = removeFaceAra(frame, cascade)

    fgmask = fgbg.apply(blur, learningRate=0)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    fgmask = cv.morphologyEx(fgmask, cv.MORPH_CLOSE, kernel, 2)

    # height, width = frame.shape[:2]
    # for x1, y1, x2, y2 in rect:
    #     cv.rectangle(fgmask, (x1 - 10, 0), (x2 + 10, height), (0, 0, 0), -1)

    img_result = process(frame, fgmask, debug=False)

    cv.imshow('mask', fgmask)
    cv.imshow('result', img_result)

    key = cv.waitKey(30) & 0xff
    if key == 27:
        break

cap.release()
cv.destroyAllWindows()