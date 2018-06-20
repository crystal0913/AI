# coding=utf-8

import subprocess
import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

# http://blog.csdn.net/liangyuannao/article/details/8896563


def disk_report():
    p = subprocess.Popen("df -h", shell=True, stdout=subprocess.PIPE)
    print(p.stdout.readlines())
    return p.stdout.readlines()


def create_pdf(input, output="disk_report.pdf"):
    now = datetime.datetime.today()
    date = now.strftime("%h %d %Y %H:%M:%S")
    c = canvas.Canvas(output)
    textobject = c.beginText()
    textobject.setTextOrigin(inch, 11*inch)
    textobject.textLines('''Disk Capcity Report: %s''' %date)
    for line in input:
        textobject.textLine(line.strip())
    c.drawText(textobject)
    # c.drawString(0,0,"asbcd")
    c.drawImage("SC201512L.png", 0, 0, width=400, height=200)
    c.showPage()
    c.drawImage("SC201306W.png", 0, 0, width=400, height=200)
    c.save()

if __name__ == "__main__":
    report = disk_report()
    create_pdf(["a", "b", "c"])