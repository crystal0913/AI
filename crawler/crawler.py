# coding=utf-8

# 网页图片爬取
import urllib.request
import urllib
import re


def gethtml(url):
    page = urllib.request.urlopen(url)
    html1 = page.read()
    return html1


def getimage(site):
    reg = 'src="(.+?\.jpg)" alt='
    imglist = re.findall(reg, site)
    print(len(imglist))
    x = 0
    for imgurl in imglist:
        urllib.request.urlretrieve(imgurl, '%s.jpg' % x)
        x += 1


def getswf():
    i = 1
    while i < 10:
        urllib.request.urlretrieve("http://tbm.alicdn.com/YlI1t0Q14T5TG33lNgp/mBLaXnwXWpm7pWJgsSo@@ld-0000"
                                   + str(i) + ".ts", str(i) + '.ts')
        i += 1

if __name__ == "__main__":
    # html = gethtml('http://pic.yxdown.com/list/0_0_1.html')
    # print(html.decode('UTF-8'))
    # #
    # print(getimage(html.decode('UTF-8')))
    getswf()
