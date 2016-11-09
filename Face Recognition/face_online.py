# Face Recognition Demo Code that runs on ubuntu
# Dermot Rossiter
# Philip Walsh

# USAGE
# python face_online.py faces/ cascades/haarcascade_frontalface_default.xml

import os
import sys
import cv2
import numpy as np
 
import random
rand = random.Random()
 
def read_images(path, sz=None):
    #path: Path to a folder with subfolders representing the subjects (persons).
    #sz: A tuple with the size Resizes
    c = 0
    X,y,z = [], [], []
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                try:
                    im = cv2.imread(os.path.join(subject_path, filename), cv2.IMREAD_GRAYSCALE)
                    if (len(im)==0):
                        continue # not an image                        
                    # resize to given size (if given)
                    if (sz is not None):
                        im = cv2.resize(im, sz)
                    X.append(np.asarray(im, dtype=np.uint8))
                    #X: The images, which is a Python list of numpy arrays.
                    y.append(c)
                    #y: The corresponding labels in a Python list.
                except (IOError, (errno, strerror)):
                    print ("I/O error({0}): {1}".format(errno, strerror))
                except:
                    print ("Unexpected error:", sys.exc_info()[0])
                    raise
            c = c+1
            z.append(subdirname)
            #z: A list of person-names, indexed by label
    return [X,y,z]

# reload the images & retrain the model.
def retrain( imgpath, model,sz ) :
    X,y,names = read_images(imgpath,sz)
    if len(X) == 0:
        print ("image path empty", imgpath)
        return [[],[],[]]
    # np.asarray to turn them into NumPy lists
    model.train(np.asarray(X), np.asarray(y, dtype=np.int32))
    return [X,y,names]

def menu():
    print ("----------------------------------------------------")
    print ("MENU")
    print ("----------------------------------------------------")
    print ("1. Recognise Faces")
    print ("2. Add User")
    print ("3. Delete User")
    print ("4. List Users")
    print ("5. Exit") 
 
 
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print ("USAGE: facerec_online.py </path/to/images> <path/to/cascadefile>")
        sys.exit()

     
    # create the img folder, if nessecary
    imgdir = sys.argv[1]
    try:
        os.mkdir(imgdir)
    except:
        pass # dir already existed
 
    # default face size, all faces in the db need to be the same.
    face_size=(90,90)
     
    # open the webcam
    cam = cv2.VideoCapture(0)
    if ( not cam.isOpened() ):
         print ("no camera found!")
         sys.exit()      
    print ("camera: ok.")      
     
    # load the cascadefile:
    cascade = cv2.CascadeClassifier(sys.argv[2])
    if ( cascade.empty() ):
         print ("No Cascade!")
         sys.exit()         
    print ("Cascade:",sys.argv[2])
     
    # Create the model.
    #model = cv2.createEigenFaceRecognizer()
    #model = cv2.createFisherFaceRecognizer()
    model = cv2.face.createLBPHFaceRecognizer()


     
    # train it from faces in the /face directory
    images,labels,names = retrain(imgdir,model,face_size)
    print ("trained:",len(images),"images",len(names),"persons")

    menu()
     
    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        # try to detect a face in the img:
        rects = cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30), flags = cv2.CASCADE_SCALE_IMAGE)       
         
        # roi will keep the cropped face image
        roi = None
        for x, y, w, h in rects:
            # crop & resize it 
            roi = cv2.resize( gray[y:y+h, x:x+h], face_size )
            # draw rectangle around face
            cv2.rectangle(img, (x,y),(x+w,y+h), (255, 0, 0))
            if len(images)>0:
                # model.predict returns the predicted label and the associated confidence
                [p_label, p_confidence] = model.predict(np.asarray(roi))
                name = "unknown"
                if p_confidence < 100 : name = names[p_label]
                cv2.putText( img, "%s %.2f" % (name, p_confidence),(x+10,y+20), cv2.FONT_HERSHEY_PLAIN,1.3, (0,200,0))
            break
 
        cv2.imshow('facedetect', img)
 
        key_pressed = cv2.waitKey(5) & 0xFF 
    
        # retrain the model when '1' is pressed
        if (key_pressed == 49):
            images,labels,names = retrain(imgdir,model,face_size)
            print ("Trained:", len(images), "Images", len(names), "People")
        
        # add person to the database when '2' is pressed
        if (key_pressed == 50) and (roi!=None): 
            print ("Please input the users name: ")
            name = sys.stdin.readline().strip('\r').strip('\n')
            # make a folder to hold images of the person
            dirname = os.path.join(imgdir,name)
            try:
                os.mkdir(dirname)
            except:
                pass
            # save image
            path=os.path.join(dirname,"%d.png" %(rand.uniform(0,10000)))
            print ("New picture added:",path)
            menu()
            cv2.imwrite(path, roi)
            
            
        # delete person from the database when '3' is pressed
        if (key_pressed == 51): 
            print ("Please input name of person to be deleted: ")
            name = sys.stdin.readline().strip('\r').strip('\n')
            dirname = os.path.join(imgdir, name)
            users = os.listdir(imgdir)
            str(name)
            for user in users:
                str(user)
                if (name == user):
                    exists = True
                else:
                    exists = False
            if (exists == True):
                fileList = os.listdir(dirname)
                for fileName in fileList:
                    os.remove(dirname + "/" + fileName)
                try:
                    os.rmdir(dirname)
                    # print deleted folder to the command line
                    print ("Deleted:",name)
                except:
                    print("Not deleted")
                    pass
            else:
                print("No user exists")

        #list all users when '4' is pressed
        if (key_pressed == 52):
            dirname = os.listdir(imgdir)
            print("------")
            print("User List:")
            print("------")
            for users in dirname:
                print (users)
            print("------")      
            
        # exit when user presses '5'
        if key_pressed == 53: break 