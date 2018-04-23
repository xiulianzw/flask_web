import pickle
import sqlite3

import numpy as np
from flask import Flask,render_template,request
from wtforms import Form,TextAreaField,validators

from chapter09.flask_web.vectorizer import vect

#创建一个falsk对象
app = Flask(__name__)
#加载分类模型
clf = pickle.load(open("pkl/classifier.pkl","rb"))

#创建一个评论数据库，在app.py运行之前先运行这个方法
def create_review_db():
    conn = sqlite3.connect("db/move_review.db")
    c = conn.cursor()
    #move_review主要包括四个字段,review_id(评论ID,主键自增)、review(评论内容)、sentiment(评论类别)、review_date(评论日期)
    c.execute("CREATE TABLE move_review (review_id INTEGER PRIMARY KEY AUTOINCREMENT,review TEXT"
              ",sentiment INTEGER,review_date TEXT)")
    conn.commit()
    conn.close()

#将评论保存到数据库中
def save_review(review,label):
    conn = sqlite3.connect("db/move_review.db")
    c = conn.cursor()
    #向数据库中插入评论
    c.execute("INSERT INTO move_review (review,sentiment,review_date) VALUES "
              "(?,?,DATETIME('now'))",(review,label))
    conn.commit()
    conn.close()

#获取评论的分类结果
def classify_review(review):
    label = {0:"negative",1:"positive"}
    #将评论转换成为特征向量
    X = vect.transform(review)
    #获取评论整数类标
    Y = clf.predict(X)[0]
    #获取评论的字符串类标
    label_Y = label[Y]
    #获取评论所属类别的概率
    proba = np.max(clf.predict_proba(X))
    return Y,label_Y,proba


#跳转到用户提交评论界面
@app.route("/")
def index():
    #验证用户输入的文本是否有效
    form = ReviewForm(request.form)
    return render_template("index.html",form=form)

#跳转到评论分类结果界面
@app.route("/main",methods=["POST"])
def main():
    form = ReviewForm(request.form)
    if request.method == "POST" and form.validate():
        #获取表单提交的评论
        review_text = request.form["review"]
        #获取评论的分类结果,类标、概率
        Y,lable_Y,proba = classify_review([review_text])
        #将概率保存2为小数并转换成为百分比的形式
        proba = float("%.4f"%proba) * 100
        #将分类结果返回给界面进行显示
        return render_template("reviewform.html",review=review_text,Y=Y,label=lable_Y,probability=proba)
    return render_template("index.html",form=form)

#用户感谢界面
@app.route("/tanks",methods=["POST"])
def tanks():
    #判断用户点击的是分类正确按钮还是错误按钮
    btn_value = request.form["feedback_btn"]
    #获取评论
    review = request.form["review"]
    #获取评论所属类标
    label_temp = int(request.form["Y"])
    #如果正确，则类标不变
    if btn_value == "Correct":
        label = label_temp
    else:
        #如果错误,则类标相反
        label = 1 - label_temp

    save_review(review,label)
    return render_template("tanks.html")

class ReviewForm(Form):
    review = TextAreaField("",[validators.DataRequired()])

if __name__ == "__main__":
    #启动服务
    app.run()