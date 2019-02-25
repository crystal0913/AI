#coding=utf-8

#!/usr/bin/python

# 导入requests库

import requests
import time

# 导入文件操作库

import os

import bs4

from bs4 import BeautifulSoup

import sys

import importlib

importlib.reload(sys)

# 给请求指定一个请求头来模拟chrome浏览器

global headers
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36'}

global count
count = 0

# 定义存储位置

global save_path
save_path = 'vw/mt/'


# 创建文件夹
def createFile(file_path):
    if os.path.exists(file_path) is False:
        os.makedirs(file_path)
    # 切换路径至上面创建的文件夹
    os.chdir(file_path)


def download(path, page_no):
    global headers
    res_sub = requests.get(path, headers=headers)
    # 解析html
    soup_sub = BeautifulSoup(res_sub.text, 'html.parser')
    # 获取页面的栏目地址
    all_a = soup_sub.find('div', class_='box list channel max-border list-text-my').find_all('a', target='_blank')
    for a in all_a:
        href = a.attrs['href']
        res_sub_1 = requests.get("https://www..com" + href, headers=headers)
        print("https://www..com" + href)
        #time.sleep(5)
        soup_sub_1 = BeautifulSoup(res_sub_1.text, 'html.parser')
        imgs = soup_sub_1.find('div', class_='content').find_all('img')
        for img in imgs:
            url = img.attrs['data-original']
            try:
                headers = {'Referer': href}
                img = requests.get(url, headers=headers)
                global count
                count += 1
                file_name = "{}.jpg".format(count)
                f = open(file_name, 'wb')
                f.write(img.content)
                f.close()
            except Exception as e:
                print(e)


if __name__ == "__main__":
    createFile(save_path)
    path = ".html"
    download(path, 1)
