# coding=utf-8
from reportlab.platypus import Paragraph, SimpleDocTemplate, Image
from reportlab.lib.styles import getSampleStyleSheet
import os
import PIL

# http://blog.csdn.net/kingken212/article/details/47209791
# http://blog.csdn.net/liangyuannao/article/details/8896563


def create_pic_pdf(filename):
    stylesheet = getSampleStyleSheet()
    normalStyle = stylesheet['Normal']
    story = []
    height = 0
    width = 0
    rootdir = "res/"
    filelist = os.listdir(rootdir)
    story.append(Paragraph("Hello, " + filename + "! count:" + str(len(filelist)), normalStyle))
    for i in range(0, len(filelist)):
        qid = filelist[i][0:-4]
        story.append(Paragraph(str(i+1) + ". " + qid, normalStyle))
        img = PIL.Image.open(rootdir + filelist[i])
        width = img.size[0] / 3
        height = img.size[1] / 3
        story.append(Image(rootdir + filelist[i], width=width, height=height))

    doc = SimpleDocTemplate(filename + ".pdf")
    doc.build(story)


def create_by_qid(qids, img_path, file_name):
    stylesheet = getSampleStyleSheet()
    normalStyle = stylesheet['Normal']
    story = []
    story.append(Paragraph("count:" + str(len(qids)), normalStyle))
    i = 1
    for qid in qids:
        # name = qid[:-1]
        name = qid
        png_name = img_path + name + ".png"
        if not os.path.exists(png_name):
            print(name)
            continue
        story.append(Paragraph(str(i) + ". " + name, normalStyle))
        img = PIL.Image.open(png_name)
        width = img.size[0] / 3
        height = img.size[1] / 3
        story.append(Image(png_name, width=width, height=height))
        i += 1
    doc = SimpleDocTemplate(file_name + ".pdf")
    doc.build(story)


if __name__ == "__main__":
    f = open("C:\\Users\\hp\\Desktop\\集合bug.txt", "r", encoding="UTF-8")
    create_by_qid(f.readlines(), "E:\\Test_Set\\jira_png\\", "pdf\\erxiangshi_bug")
