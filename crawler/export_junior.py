import os
import shutil
import time

import creadepdf2
from selenium import webdriver
from PIL import Image


def capture(qids):
    browser = webdriver.Firefox(executable_path='geckodriver.exe')
    # browser.set_window_size(1200, 600)
    count = len(qids)
    if count < 1:
        return
    browser.get(httpStr)  # Load page
    if need_std:
        browser.find_element_by_id("showCont").click()
    for qId in qids:
        count -= 1
        if os.path.exists(pic_path + qId + ".png"):
            continue
        try:
            browser.find_element_by_id("questionidinput").clear()
            browser.find_element_by_id("questionidinput").send_keys(qId)  # input qid
            browser.find_element_by_class_name("btn-purple").click()
            time.sleep(2)
            if "查无此题" in browser.find_element_by_id("question_stem").text:
                print(qId + "  no this id")
                continue
            save_fn = qId + ".png"
            # browser.save_screenshot("res/" + save_fn)
            img1_name = pic_path + "stem" + save_fn
            img2_name = pic_path + "sub" + save_fn

            browser.find_element_by_id("question_stem").screenshot(img1_name)
            browser.find_element_by_id("question_substem").screenshot(img2_name)

            img1 = Image.open(img1_name)
            img2 = Image.open(img2_name)
            if need_std:
                img4_name = pic_path + "std" + save_fn
                browser.find_element_by_id("stdAnsHtml").screenshot(img4_name)
                img4 = Image.open(img4_name)
                new_img = images_joint([img1, img2, img4])
                os.remove(img4_name)
            else:
                new_img = images_joint([img1, img2])
            os.remove(img1_name)
            os.remove(img2_name)
            new_img.save(pic_path + save_fn)
            print(count)
        except Exception as e:
            print(e)
            continue
    browser.close()


# 图片合并
def images_joint(images):
    new_height = 0
    new_width = images[0].size[0]
    for img in images:
        new_height += img.size[1]
    new_img = Image.new('RGB', (new_width, new_height), (241, 241, 241))
    cur_height = 0
    for img in images:
        new_img.paste(img, (0, cur_height))
        cur_height += img.size[1]
    return new_img


def getqids(lines):
    qids = []
    for line in lines:
        qids.append(line[0:getid_endpos(line)])
    return qids


def getid_endpos(line):
    return len(line) - 1
    #
    # for i in range(len(line)):
    #     if line[i].isspace() or '\u4e00' <= line[i] <= '\u9fa5':
    #         break
    # return i


def delete_dir(rootdir):
    filelist = os.listdir(rootdir)
    for f in filelist:
        filepath = os.path.join(rootdir, f)
        if os.path.isfile(filepath):
            os.remove(filepath)
        elif os.path.isdir(filepath):
            shutil.rmtree(filepath, True)
    shutil.rmtree(rootdir, True)


httpStr = "http://"
pic_path = "E:/q_png_junior/"
pdf_path = "E:/exportpdf/"  # 输出pdf文件的目录
need_std = True

if __name__ == "__main__":
    qid_file_path = "test.txt"  # 题目id文本文件
    pdf_name = "pdfname"
    f = open(qid_file_path, "r", encoding="UTF-8")
    qids = getqids(f.readlines())
    for qid in qids:
        print(qid)
    print(len(qids))
    if not os.path.isdir(pic_path):
        os.mkdir(pic_path)
    if not os.path.isdir(pdf_path):
        os.mkdir(pdf_path)
    capture(qids)
    creadepdf2.create_by_qid(qids, pic_path, pdf_path + "/" + pdf_name)
    delete_dir(pic_path)
