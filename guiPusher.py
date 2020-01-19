from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os

from tkinter import *
from tkinter import messagebox
import time

from flask import Flask, request, render_template, Response
import datetime as Hari
from datetime import datetime
import webbrowser
import numpy as np
import face_recognition
from pyimagesearch.motion_detection import SingleMotionDetector
import threading
from threading import Timer
import argparse
import imutils

from imutils.video import VideoStream
from imutils.video import FPS

import sqlite3
from subprocess import Popen
import signal # to kill pid (signal.SIGTERM)

import pusher

pusher_client = pusher.Pusher(
    '883694',
    '58eec546d8492ceb70ab',
    '88d3560fb18da495a9ef',
    ssl=True,
    cluster='ap1')


def trainManual():
    root= Tk()
    root.title("Register New (Manual)")

    canvas1 = Canvas(root, width = 400, height = 400,  relief = 'raised')
    canvas1.pack()

    label1 = Label(root, text='Register New User')
    label1.config(font=('helvetica', 14))
    canvas1.create_window(200, 25, window=label1)

    label2 = Label(root, text='Enter your ID:')
    label2.config(font=('helvetica', 10))
    canvas1.create_window(200, 100, window=label2)

    entry1 = Entry (root)
    canvas1.create_window(200, 120, window=entry1) 

    label3 = Label(root, text='Enter your name:')
    label3.config(font=('helvetica', 10))
    canvas1.create_window(200, 150, window=label3)

    entry2 = Entry (root)
    canvas1.create_window(200, 170, window=entry2) 

    label5 = Label(root, text='Enter user position:')
    label5.config(font=('helvetica', 10))
    canvas1.create_window(200, 200, window=label5)

    entry3 = Entry (root)
    canvas1.create_window(200, 220, window=entry3) 

    def submitData ():
        
        name = entry2.get()
        folderName = name                                                       # creating the person or user folder
        folderPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "dataset/"+folderName)
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)

        messagebox.showinfo("Information","Please place minimum 3 images in your folder '" +name +"' in dataset folder before pressing Train Image.")
        button1 = Button(root,text='Train Image', command=trainImage, bg='brown', fg='white', width = 10, font=('helvetica', 9, 'bold'))
        canvas1.create_window(200, 260, window=button1)

    def trainImage ():
        
        x1 = entry1.get()
        ap = argparse.ArgumentParser()
        ap.add_argument("-i", "--dataset", default='dataset')
        ap.add_argument("-e", "--encodings", default='data/encodings.pickle')
        ap.add_argument("-d", "--detection-method", type=str, default="hog")
        args = vars(ap.parse_args())

        root.update()
        label4 = Label(root, text= "[INFO] quantifying faces...",font=('helvetica', 10, 'bold'))
        canvas1.create_window(200, 300, window=label4)
        
        imagePaths = list(paths.list_images(args["dataset"]))

        knownEncodings = []
        knownNames = []

        for (i, imagePath) in enumerate(imagePaths):
            root.update()
            label4 = Label(root, text= "[INFO] processing image {}/{}".format(i + 1,
                len(imagePaths)),font=('helvetica', 10, 'bold'))
            canvas1.create_window(200, 320, window=label4)

            name = imagePath.split(os.path.sep)[-2]
            image = cv2.imread(imagePath)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #GRB to dlib ordering (RGB)

            boxes = face_recognition.face_locations(rgb,
                model=args["detection_method"]) # detect face in image

            encodings = face_recognition.face_encodings(rgb, boxes)

            for encoding in encodings:
                knownEncodings.append(encoding)
                knownNames.append(name)

        root.update()
        label4 = Label(root, text= "[INFO] serializing encodings...",font=('helvetica', 10, 'bold'))
        canvas1.create_window(200, 340, window=label4)
        data = {"encodings": knownEncodings, "names": knownNames}
        f = open(args["encodings"], "wb")
        f.write(pickle.dumps(data))
        f.close()

        root.update()
        label4 = Label(root, text= "Training Data Complete",font=('helvetica', 10, 'bold'))
        canvas1.create_window(200, 360, window=label4)
        root.update()

        idUser = entry1.get()
        name = entry2.get()
        position = entry3.get()

        conn = sqlite3.connect('data/database.db')
        conn.row_factory = dict_factory
        curr = conn.cursor()
        statusReset = "No"

        curr.execute("INSERT INTO names (staffID, nameA, positionA,status) VALUES ('"+ str(idUser) +"','"+ str(name)+"','"+str(position)+"','No');")
        conn.commit()

        messagebox.showinfo("Information","Training Done!")
        button2 = Button (root, text="Close Windows", command=root.destroy,  width=13,bg='brown', fg='white', font=('helvetica', 9, 'bold'))
        canvas1.create_window(200, 260, window=button2)

    button1 = Button(root,text='Submit', command=submitData, bg='brown', width=10, fg='white', font=('helvetica', 9, 'bold'))
    canvas1.create_window(200, 260, window=button1)

    root.mainloop()


def registerNew():
    root= Tk()
    root.title("Register New")

    canvas1 = Canvas(root, width = 400, height = 400,  relief = 'raised')
    canvas1.pack()

    label1 = Label(root, text='Register New User')
    label1.config(font=('helvetica', 14))
    canvas1.create_window(200, 25, window=label1)

    label2 = Label(root, text='Enter your ID:')
    label2.config(font=('helvetica', 10))
    canvas1.create_window(200, 100, window=label2)

    entry1 = Entry (root)
    canvas1.create_window(200, 120, window=entry1) 

    label3 = Label(root, text='Enter your name:')
    label3.config(font=('helvetica', 10))
    canvas1.create_window(200, 150, window=label3)

    entry2 = Entry (root)
    canvas1.create_window(200, 170, window=entry2) 

    label5 = Label(root, text='Enter user position:')
    label5.config(font=('helvetica', 10))
    canvas1.create_window(200, 200, window=label5)

    entry3 = Entry (root)
    canvas1.create_window(200, 220, window=entry3) 

    def captureImage ():
        faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        clear = lambda: os.system('cls') # windows
        #clear = lambda: os.system('clear') # linux
        clear()
        print('\n')
        name = entry2.get()

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 512)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 512)

        folderName = name                                                       # creating the person or user folder
        folderPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "dataset/"+folderName)
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)

        sampleNum = 0
        messagebox.showinfo("Information", "System will take 6 images for the training")
        while sampleNum <= 5:
            ret, img = cap.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5,minSize=(100, 100))

            for (x, y, w, h) in faces:
                sampleNum += 1
                cv2.imwrite("dataset/" + name +"/"+ str(sampleNum)+'.jpg', img, [int(cv2.IMWRITE_PNG_COMPRESSION), 2])
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
                
#                 time.sleep(1)

#                 messagebox.showinfo("Information", "Move around your face")

            cv2.imshow('Capture Photo', img)                                                    # showing the video input from camera on window
            cv2.waitKey(1)
            
        cap.release()                                                                   # turning the webcam off
        cv2.destroyAllWindows()

        messagebox.showinfo("Information","Images successfully taken!")
        button1 = Button(root,text='Submit', command=trainImage, bg='brown', fg='white', width = 10, font=('helvetica', 9, 'bold'))
        canvas1.create_window(200, 260, window=button1)

    def trainImage ():
        
        x1 = entry1.get()
        ap = argparse.ArgumentParser()
        ap.add_argument("-i", "--dataset", default='dataset')
        ap.add_argument("-e", "--encodings", default='data/encodings.pickle')
        ap.add_argument("-d", "--detection-method", type=str, default="hog")
        args = vars(ap.parse_args())

        root.update()
        label4 = Label(root, text= "[INFO] quantifying faces...",font=('helvetica', 10, 'bold'))
        canvas1.create_window(200, 300, window=label4)
        
        imagePaths = list(paths.list_images(args["dataset"]))

        knownEncodings = []
        knownNames = []

        for (i, imagePath) in enumerate(imagePaths):
            root.update()
            label4 = Label(root, text= "[INFO] processing image {}/{}".format(i + 1,
                len(imagePaths)),font=('helvetica', 10, 'bold'))
            canvas1.create_window(200, 320, window=label4)

            name = imagePath.split(os.path.sep)[-2]
            image = cv2.imread(imagePath)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #GRB to dlib ordering (RGB)

            boxes = face_recognition.face_locations(rgb,
                model=args["detection_method"]) # detect face in image

            encodings = face_recognition.face_encodings(rgb, boxes)

            for encoding in encodings:
                knownEncodings.append(encoding)
                knownNames.append(name)

        root.update()
        label4 = Label(root, text= "[INFO] serializing encodings...",font=('helvetica', 10, 'bold'))
        canvas1.create_window(200, 340, window=label4)
        data = {"encodings": knownEncodings, "names": knownNames}
        f = open(args["encodings"], "wb")
        f.write(pickle.dumps(data))
        f.close()

        root.update()
        label4 = Label(root, text= "Training Data Complete",font=('helvetica', 10, 'bold'))
        canvas1.create_window(200, 360, window=label4)
        root.update()

        idUser = entry1.get()
        name = entry2.get()
        position = entry3.get()

        conn = sqlite3.connect('data/database.db')
        conn.row_factory = dict_factory
        curr = conn.cursor()
        statusReset = "No"

        curr.execute("INSERT INTO names (staffID, nameA, positionA,status) VALUES ('"+ str(idUser) +"','"+ str(name)+"','"+str(position)+"','No');")
        conn.commit()

        messagebox.showinfo("Information","Training Done!")
        button2 = Button (root, text="Close Windows", command=root.destroy,  width=13,bg='brown', fg='white', font=('helvetica', 9, 'bold'))
        canvas1.create_window(200, 260, window=button2)

    button1 = Button(root,text='Take Picture', command=captureImage, bg='brown', width=10, fg='white', font=('helvetica', 9, 'bold'))
    canvas1.create_window(200, 260, window=button1)

    root.mainloop()
    

lock = threading.Lock()
outputFrame = None

def dict_factory(cursor, row):
    d ={}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d

def Enquiry(lis1): 
    if len(lis1) == 0: 
        return 0
    else: 
        return 1

running = None # Global flag

def start():
    """ Enable the flask test by setting the global flag to True"""
    global running
    running = True

# def faceRecognition ():
    if running:
        app = Flask(__name__)
        
        
        print("[INFO] starting video stream...")
        vs = VideoStream(src=0).start()

        def detect_motion(frameCount):

            frame_number = 0
            global cap, outputFrame, lock, data

            md = SingleMotionDetector(accumWeight=0.1)
            total = 0
            print("[INFO] loading encodings + face detector...")
            detector = cv2.CascadeClassifier(args["cascade"])

            fps = FPS().start()

            conn = sqlite3.connect('data/database.db')
            conn.row_factory = dict_factory
            curr = conn.cursor()
            statusReset = "No"

            curr.execute("UPDATE names SET status = 'No';")
            conn.commit()

            while True:
                
                frame = vs.read()
                frame = imutils.resize(frame, width=500)

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # face detection
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # face recognition

                rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
                    minNeighbors=5, minSize=(30, 30),
                    flags=cv2.CASCADE_SCALE_IMAGE)

                boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]

                encodings = face_recognition.face_encodings(rgb, boxes)
                names = []

                for (top, right, bottom, left), face_encoding in zip(boxes, encodings):
                    data = pickle.loads(open(args["encodings"], "rb").read())

                    matches = face_recognition.compare_faces(data["encodings"],
                    face_encoding, tolerance=0.4)

                    face_distances = face_recognition.face_distance(data["encodings"], face_encoding)
                    best_match_index = np.argmin(face_distances)

                    if matches[best_match_index]:
                        name = data["names"][best_match_index]
                        status = True

                        namaa = str(name)

                        check_if_data_exist = "SELECT * from names WHERE nameA='" + str(name) + "';"
                        result = curr.execute(check_if_data_exist).fetchall()

                        if  result[0]['status'] =="No" :


                            name = result[0]['nameA']
                            position = result[0]['positionA']
                            dateTime = datetime.now()
                            dateSt = str(datetime.now())
                           
                            status = "Found"

                            data = {
                                "id": result[0]['id'],
                                "nameA": name,
                                "positionA": position,
                                "status": status,
                                "dateTime": dateSt
                                }

                            pusher_client.trigger('table', 'new-record', {'data': data })
                        
                            update_sql = '''UPDATE names SET status = ?, dateTime = ? WHERE nameA = ? '''
                            curr.execute(update_sql, (status, dateSt, name))
                            conn.commit()
                            dateSt = str(datetime.now()) 

                        check_if_data_exist = "SELECT * from names WHERE nameA='" + str(name) + "';"
                        result = curr.execute(check_if_data_exist).fetchall()

                        if  result[0]['status'] =="Found" :

                            name = result[0]['nameA']
                            position = result[0]['positionA']
                            dateTime = datetime.now()
                            dateSt = str(datetime.now())
                           
                            status = "Found"
                        
                            update_sql = '''UPDATE names SET status = ?, dateTime = ? WHERE nameA = ? '''
                            curr.execute(update_sql, (status, dateSt, name))
                            conn.commit()
                            dateSt = str(datetime.now())

                            check_if_data_exist = "SELECT id from names WHERE nameA='" + name + "';"
                            result = curr.execute(check_if_data_exist).fetchall()

                            data = {
                                "id": result[0]['id'],
                                "nameA": name,
                                "positionA": position,
                                "status": status,
                                "dateTime": dateSt
                                }

                            pusher_client.trigger('table', 'update-record', {'data': data })     

                        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
                        font = cv2.FONT_HERSHEY_DUPLEX
                        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
                        face_image = frame[top:bottom, left:right]
                        cv2.imwrite( "person_found/" + name + '_Face.jpg', face_image)          

                    else:

                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                        font = cv2.FONT_HERSHEY_DUPLEX
                        cv2.putText(frame, "Unknown", (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

                timestamp = Hari.datetime.now()
                cv2.putText(frame, timestamp.strftime(
                    "%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
                
                if total > frameCount:
                    motion = md.detect(gray)
                    if motion is not None:
                        (thresh, (minX, minY, maxX, maxY)) = motion

                md.update(gray)
                total += 1

                with lock:
                    outputFrame = frame.copy()

        def generate():
            
            global outputFrame, lock
            while True:
                with lock:
                    if outputFrame is None:
                        continue

                    (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

                    if not flag:
                        continue

                yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                    bytearray(encodedImage) + b'\r\n')

        @app.route("/video_feed")
        def video_feed():
            
            return Response(generate(),
                mimetype = "multipart/x-mixed-replace; boundary=frame")

        @app.route('/')
        def index(): 
            conn = sqlite3.connect('data/database.db')
            conn.row_factory = dict_factory
            curr = conn.cursor()
            statusReset = "No"

            curr.execute("UPDATE names SET status = 'No';")
            conn.commit()
            
            all_faces = curr.execute("SELECT * from names WHERE status = 'Found';").fetchall()       
            return render_template('facergns.html', names=all_faces)


        @app.route('/resetAll')
        def resetAll():               
            conn = sqlite3.connect('data/database.db')
            conn.row_factory = dict_factory
            curr = conn.cursor()
            statusReset = "No"

            curr.execute("UPDATE names SET status = 'No';")
            conn.commit()
            all_faces = curr.execute("SELECT * from names WHERE status = 'Found';").fetchall() 
                   
            return render_template('facergns.html', names=all_faces)

        @app.route("/shutdown")
        def shutdown():
            Popen('python guiPusher.py', shell='True')
            # Popen('python3 guiPusher.py', shell='False') # in linux/raspberry pi use this line
            pid = os.getpid()
            print('Back to main GUI...')
            os.kill(int(pid), signal.SIGINT)
            # os.kill(int(pid), signal.SIGKILL) # in linux/raspberry pi use this line
    
        def open_browser():
              webbrowser.open_new('http://127.0.0.1:5000/')

        if __name__ == '__main__':
            
            ap = argparse.ArgumentParser()
            ap.add_argument("-f", "--frame-count", type=int, default=32,
                help="# of frames used to construct the background model")
            ap.add_argument("-c", "--cascade", default='data/haarcascade_frontalface_default.xml')
            ap.add_argument("-e", "--encodings", default='data/encodings.pickle')
            args = vars(ap.parse_args())
            args = vars(ap.parse_args())

            
            t = threading.Thread(target=detect_motion, args=(args["frame_count"],))
            t.daemon = True
            t.start()
            #app.run(debug=True, threaded=True, use_reloader=False, host='192.168.0.132', port=5000)
            Timer(1, open_browser).start();
            app.run(debug=True, threaded=True, use_reloader=False)

#     gui.after(1000, faceRecognition) # after 1 second, call the flaskTest again (creating a recursive loop)

if __name__ == "__main__": 
    gui = Tk() 

    gui.configure(background="black") 
    gui.title("Main Menu") 
    canvas2 = Canvas(gui, width = 400, height = 500,  relief = 'raised', bg="white")
    canvas2.pack()

    photo = PhotoImage(file='static/img/facergns-logo-synapse@2x.png')
    label = Label(gui, image=photo, bd=0)
    canvas2.create_window(200,50, window=label)
    
    face = Button(gui, text='Run Face Recognition', fg='white', 
        activebackground= 'red', activeforeground='black',font= "Cambria 11 bold", bg='VioletRed4', 
                command=start, height=3, width=21) 
    canvas2.create_window(200, 180, window=face)

    reg = Button(gui, highlightcolor='blue', text='Register New User \n (Camera)', font= "Cambria 11 bold", fg='white', bg='VioletRed4', 
        activebackground= 'red', activeforeground='black',
                command=registerNew, height=3, width=21) 
    canvas2.create_window(200, 260, window=reg)

    button1 = Button(gui, text='Register New User \n (Manual)', fg='white',font= "Cambria 11 bold",bg='VioletRed4', 
        activebackground= 'red', activeforeground='black',
                command=trainManual, height=3, width=21) 
    canvas2.create_window(200, 340, window=button1)

#     gui.after(1000, faceRecognition) 
    gui.mainloop() 
