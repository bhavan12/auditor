import pyodbc
from flask import Flask, render_template, send_file, make_response
import  pandas as pd
con=pyodbc.connect(
"DRIVER={SQL Server};server=10.10.10.3;database=AizantIT_ML;uid=rnd;pwd=AizantIT123")
app = Flask(__name__)
@app.route('/')
def begin():
    return "selected successfully"
