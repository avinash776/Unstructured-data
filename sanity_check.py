import sys, os, io
import json

report = {}

# Basic imports
try:
    import numpy as np
    import PIL
    from PIL import Image
    report['numpy'] = np.__version__
    report['Pillow'] = PIL.__version__
except Exception as e:
    report['numpy/PIL_error'] = str(e)

# OpenCV face cascade check
try:
    import cv2
    report['cv2'] = cv2.__version__
    # create a small gray img
    import numpy as np
    img = np.zeros((64,64,3), dtype=np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Load cascade from OpenCV package if present else skip
    cascade_path = 'haarcascade_frontalface_default.xml'
    if not os.path.exists(cascade_path):
        # try cv2.data.haarcascades
        try:
            cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
        except Exception:
            cascade_path = None
    if cascade_path and os.path.exists(cascade_path):
        face_cascade = cv2.CascadeClassifier(cascade_path)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        report['opencv_faces_detected'] = int(len(faces))
    else:
        report['opencv_cascade'] = 'missing'
except Exception as e:
    report['cv2_error'] = str(e)

# YOLO object detection removed
report['ultralytics'] = 'removed'

# rembg smoke test
try:
    from rembg import remove
    import numpy as np
    arr = np.full((32,32,3), 255, dtype=np.uint8)
    out = remove(arr)
    report['rembg'] = type(out).__name__
except Exception as e:
    report['rembg_error'] = str(e)

# DeepFace minimal analyze with enforce_detection=False
try:
    from deepface import DeepFace
    img = Image.new('RGB', (64,64), color='white')
    img.save('tmp_df.jpg')
    analysis = DeepFace.analyze('tmp_df.jpg', actions=['age','gender','emotion'], enforce_detection=False)
    report['deepface'] = 'ok'
    try:
        os.remove('tmp_df.jpg')
    except Exception:
        pass
except Exception as e:
    report['deepface_error'] = str(e)

print(json.dumps(report, indent=2))
