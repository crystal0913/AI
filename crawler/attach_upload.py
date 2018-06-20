# coding=utf-8
import re
import time
import os
import urllib.request

import printscreen as ps
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException

jira_url = "http://"
user_agent = 'Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)'
headers = {'User-Agent': user_agent}
pic_dir = "jira_png\\"


def getqid_dict(file_path):
    url2id = {}
    f = open(file_path, "r", encoding="UTF-8")
    lines = f.readlines()
    for line in lines:
        sp = line[0:len(line) - 1].split("###")
        url2id.setdefault(sp[0], sp[1][0:ps.getid_endpos(sp[1])])
    return url2id


def upload(url2id_dict):
    browser = webdriver.Firefox(executable_path=r'D:\Program Files\geckodriver-v0.15.0-win64\geckodriver.exe')

    # 设置等待时间
    browser.implicitly_wait(3)
    # login
    browser.get("http://172.16.3.11:10000/login.jsp")
    browser.implicitly_wait(3)
    browser.find_element_by_id("login-form-username").send_keys("chenxianling")
    browser.find_element_by_id("login-form-password").send_keys("123456")
    browser.find_element_by_id("login-form-submit").click()
    time.sleep(5)

    i = len(url2id_dict)
    for k, v in url2id_dict.items():
        try:
            png_name = pic_dir + v + ".png"
            if not os.path.exists(png_name):
                continue
            browser.get(jira_url + k)
            browser.implicitly_wait(3)
            if isElementExist(browser, "attachmentmodule"):  # 是否已经存在附件
                print("exist")
                continue
            browser.find_element_by_id("opsbar-operations_more").click()
            browser.find_element_by_id("attach-file").click()
            upfile = browser.find_element_by_class_name("upfile")
            upfile.send_keys(png_name)
            time.sleep(1)
            browser.find_element_by_id("attach-file-submit").click()
            print(str(i) + "  " + k + "  " + v)
            i -= 1
            time.sleep(1)
        except Exception as e:
            print(e)
            print(str(i) + "  " + k + "  " + v + "ERROR")
            continue


def isElementExist(driver, element):
    try:
        driver.find_element_by_id(element)
        return True
    except NoSuchElementException:
        return False


def get_all_url():

    # 列表视图
    url = 'http://'
    i = 0
    total = 200
    count = 0
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
            print(t.get("title"))
            count += 1
            # print(t.get("data-key") + "###" + t.get("title"))
        i += 50
    print(count)


def get_all_url2():
    url = 'http://'
    req = urllib.request.Request(url, headers=headers)
    myResponse = urllib.request.urlopen(req)
    myPage = myResponse.read()
    unicodePage = myPage.decode("utf-8")
    soup = BeautifulSoup(unicodePage, "html.parser")
    # print(soup.contents)
    # tt = soup.select('li[data-key = "AUTOSOLVE-%s"]')
    tt = soup.find_all('li', attrs={'data-key': re.compile('SOLVE-.*')})
    for t in tt:
        print(t.get("data-key") + "###" + t.get("title"))
        # print(t.get("title"))


if __name__ == "__main__":
    # dicts = getqid_dict("C:\\Users\\hp\\Desktop\\t1.txt")
    # upload(dicts)
    get_all_url()
