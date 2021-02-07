import cv2

reye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')
leye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')

cap = cv2.VideoCapture(0)
for i in range(500):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    right_eye =  reye.detectMultiScale(gray)
    left_eye = leye.detectMultiScale(gray)
    
    for (x,y,w,h) in right_eye:
        r_eye = frame[y:y+h,x:x+w]
        r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye,(24,24))
        cv2.imwrite('Datasets/Closed/'+str(i)+'right.png', r_eye)
        break
        
    for (x,y,w,h) in left_eye:
        l_eye = frame[y:y+h,x:x+w]
        l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)
        l_eye = cv2.resize(l_eye,(24,24))
        cv2.imwrite('Datasets/Closed/'+str(i)+'left.png', l_eye)
        break
del(cap)