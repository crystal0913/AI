# -- coding: utf-8 --

import os
import shutil
import time
import urllib.request
import re

import creadepdf2
from selenium import webdriver
from PIL import Image
from bs4 import BeautifulSoup

httpStr = "http://"
pic_dir = "\\stem_screenshot\\"
headers = {'User-Agent': 'Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)'}


def capture(url):
    browser = webdriver.Firefox(executable_path=r'D:\Program Files\geckodriver-v0.15.0-win64\geckodriver.exe')
    browser.set_window_size(1200, 600)
    count = len(url)
    for qId in url:
        count -= 1
        if os.path.exists(pic_dir + qId + ".png"):
            continue
        try:
            if '-' in qId:
                browser.get(httpStr + qId[:-2])  # Load page
            else:
                browser.get(httpStr + qId)  # Load page
            if "查无此题" in browser.find_element_by_id("question_stem").text:
                print(qId + "  no this id")
                continue
            browser.find_element_by_id("showCont").click()  # 模拟点击
            time.sleep(2)
            save_fn = qId + ".png"

            # browser.save_screenshot("res/" + save_fn)
            img1_name = pic_dir + "a" + save_fn
            img2_name = pic_dir + "b" + save_fn
            browser.find_element_by_id("question_stem").screenshot(img1_name)
            browser.find_element_by_id("stdAnsHtml").screenshot(img2_name)

            img1 = Image.open(img1_name)
            img2 = Image.open(img2_name)
            new_img = image_joint(img1, img2)
            os.remove(img1_name)
            os.remove(img2_name)
            new_img.save(pic_dir + save_fn)
            print(count)
        except Exception as e:
            print(e)
            continue
    browser.close()


def image_joint(image1, image2):
    height1 = image1.size[1]
    width1 = image1.size[0]
    height2 = image2.size[1]
    new_img = Image.new('RGB', (width1, height1 + height2), (241, 241, 241))
    new_img.paste(image1, (0, 0))
    new_img.paste(image2, (0, height1))
    return new_img


def getid_endpos(line):
    # return len(line) - 1
    #
    for i in range(len(line)):
        if line[i].isspace() or '\u4e00' <= line[i] <= '\u9fa5':
            break
    return i


def getqids(lines):
    qids = []
    for line in lines:
        qids.append(line[0:getid_endpos(line)])
    return qids


def delete_dir(rootdir):
    filelist = os.listdir(rootdir)
    for f in filelist:
        filepath = os.path.join(rootdir, f)
        if os.path.isfile(filepath):
            os.remove(filepath)
        elif os.path.isdir(filepath):
            shutil.rmtree(filepath, True)


def split(lines):
    qids = []
    for line in lines:
        qids.append(line.split("###")[1])
    return qids


# 列表视图url
def get_qid_by_url(url):
    i = 0
    total = 200
    count = 0
    question_ids = []
    while i <= total:
        req = urllib.request.Request(url + str(i), headers=headers)
        myResponse = urllib.request.urlopen(req)
        myPage = myResponse.read()
        unicodePage = myPage.decode("utf-8")
        soup = BeautifulSoup(unicodePage, "html.parser")
        # print(soup.contents)
        # tt = soup.select('li[data-key = "AUTOSOLVE-%s"]')
        tt = soup.find_all('li', attrs={'data-key': re.compile('AUTOSOLVE-.*')})
        for t in tt:
            question_ids.append(t.get("title"))
            print(t.get("title"))
            count += 1
            # print(t.get("data-key") + "###" + t.get("title"))
        i += 50
    print(count)
    return getqids(question_ids)


if __name__ == "__main__":

    f = open("C:\\Users\\hp\\Desktop\\12.txt", "r", encoding="utf-8")
    qids = getqids(f.readlines())
    # qids = get_qid_by_url(url)
    for qid in qids:
        print(qid)
    print(len(qids))
    capture(qids)
    creadepdf2.create_by_qid(qids, pic_dir, "C:\\Users\\hp\\Desktop\\不含")
    # capture(['BJ201312', 'bj1315'])
