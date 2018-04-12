from django.shortcuts import render,HttpResponse
from django.template.loader import get_template
import json
import base64
from faceapp import ismyface
import cv2
import dlib
import numpy as np

def index(request):
	template=get_template('index.html')
	html = template.render(locals())
	return HttpResponse(html)
def index1(request):
     data=request.body.decode('utf-8')
     b64=json.loads(data).get('img64').split('base64,')[1]
     imgdata=base64.b64decode(b64) 
     nparr = np.fromstring(imgdata,np.uint8)
     detector = dlib.get_frontal_face_detector()
     img_np = cv2.imdecode(nparr,cv2.IMREAD_COLOR) 
     gray_image = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
     dets = detector(gray_image, 1)
     for i, d in enumerate(dets):
        x1 = d.top() if d.top() > 0 else 0
        y1 = d.bottom() if d.bottom() > 0 else 0
        x2 = d.left() if d.left() > 0 else 0
        y2 = d.right() if d.right() > 0 else 0
        face = img_np[x1:y1,x2:y2]
        face = cv2.resize(face, (64,64))
        json_str = json.dumps(ismyface.foo.is_my_face(face))
     return HttpResponse(json_str,content_type="application/json")