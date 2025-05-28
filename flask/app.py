from flask import Flask, request, render_template, session
from numpy import *
from cvzone import *
from pickle import load
import numpy as np
import re, cv2
import sqlite3

app = Flask(__name__)
app.secret_key = "this is very confidential"

# Database connection using SQLite
DATABASE = 'database.db'


def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


@app.route("/")
def project():
    return render_template("index.html")


@app.route("/hero")
def home():
    return render_template("index.html")


@app.route("/model")
def model():
    return render_template("model.html")


@app.route("/login")
def login():
    return render_template("login.html")


@app.route("/register")
def register():
    return render_template("register.html")


@app.route("/reg", methods=["POST", "GET"])
def signup():
    msg = ""
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        password = request.form["password"]
        sql = "SELECT * FROM REGISTER WHERE NAME = ?"
        try:
            conn = get_db()
            cur = conn.cursor()
            cur.execute(sql, (name,))
            result = cur.fetchone()
            if result:
                return render_template("login.html", error=True)
            elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
                msg = "Invalid email address"
            else:
                insert_sql = "INSERT INTO REGISTER (NAME, EMAIL, PASSWORD) VALUES (?, ?, ?)"
                cur.execute(insert_sql, (name, email, password))
                conn.commit()
                msg = "Account created successfully"
        except sqlite3.Error as e:
            msg = str(e)
        finally:
            conn.close()
    return render_template("login.html", msg=msg)


@app.route("/log", methods=["POST", "GET"])
def login1():
    msg = ""
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        sql = "SELECT * FROM REGISTER WHERE EMAIL = ? AND PASSWORD = ?"
        try:
            conn = get_db()
            cur = conn.cursor()
            cur.execute(sql, (email, password))
            account = cur.fetchone()
            print(account)
            if account:
                session["Loggedin"] = True
                session["id"] = account["EMAIL"]
                session["email"] = account["EMAIL"]
                return render_template("model.html")
            else:
                msg = "Incorrect Email / Password"
        except sqlite3.Error as e:
            msg = str(e)
        finally:
            conn.close()
    return render_template("login.html", msg=msg)


@app.route("/predict")
def predict():
    cap = cv2.VideoCapture("carParkingInput.mp4")
    with open("parkingSlotPosition", "rb") as f:
        posList = load(f)

    width, height = 167, 58

    def checkParkingSpace(imgPro):
        spaceCounter = 0

        for pos in posList:
            x, y = pos
            imgCrop = imgPro[y: y + height, x: x + width]

            count = cv2.countNonZero(imgCrop)
            if count < 900:
                color = (0, 255, 0)
                thickness = 5
                spaceCounter += 1

            else:
                color = (0, 0, 255)
                thickness = 2

            cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), color, thickness)
        putTextRect(img, f"Free : {spaceCounter}/{len(posList)}", (100, 50), scale=3, thickness=5, offset=20,
                    colorR=(0, 200, 0))

    while True:
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        success, img = cap.read()

        imgG = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgB = cv2.GaussianBlur(imgG, (3, 3), 1)
        imgT = cv2.adaptiveThreshold(imgB, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16)
        imgM = cv2.medianBlur(imgT, 5)
        kernel = np.ones((3, 3), np.uint8)
        imgD = cv2.dilate(imgM, kernel, iterations=1)
        checkParkingSpace(imgD)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) and 0xFF == ord('q'):
            break


if __name__ == "__main__":
    app.run(debug=True)
