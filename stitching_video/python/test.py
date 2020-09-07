import numpy as np
import cv2

cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
read = True
count = 0

while read:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret:
        if count == 0:
            h,w,rgb = frame.shape[:3]
            batches = np.array([])
            writer = cv2.VideoWriter('outputs/test.mp4',fourcc,fps,(w,h))
        # Display the resulting frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            read = False
            writer.release()
        batches = np.append(batches,cv2.UMat(frame))
        count += 1
        if batches.size > 100:
            for batch in batches:
                print(batch)
                writer.write(batch)
            batches = np.array([])
            
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
    

    
