# coding=utf-8
import _thread
import re
import time
import urllib
import urllib.request


# ----------- 加载处理糗事百科 -----------
class SpiderModel:
    # 声明self:含有page pages enabled
    def __init__(self):
        self.page = 2
        self.pages = []
        self.enable = False

        # 将所有的段子都扣出来，添加到列表中并且返回列表

    def GetPage(self, page):
        myUrl = "http://www.qiushibaike.com/hot/page/" + page + "/?s=4946202"
        user_agent = 'Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)'
        headers = {'User-Agent': user_agent}
        req = urllib.request.Request(myUrl, headers=headers)
        myResponse = urllib.request.urlopen(req)
        myPage = myResponse.read()
        unicodePage = myPage.decode("utf-8")
        # print(unicodePage)
        # 找出所有class="content"的div标记
        # re.S是任意匹配模式，也就是.可以匹配换行符
        myItems = re.findall('<div.*?class="content">\n\n+<span>(.*?)</span>\n\n+</div>', unicodePage, re.S)
        return myItems

    # 用于加载新的段子
    def LoadPage(self):
        # 如果用户未输入quit则一直运行
        while self.enable:
            # 如果pages数组中的内容小于2个
            # print len(self.pages)
            if len(self.pages) < 2:
                try:
                    # 获取新的页面中的段子们
                    myPage = self.GetPage(str(self.page))
                    self.page += 1
                    self.pages.append(myPage)
                except Exception as e:
                    print(e)
                    print('无法链接!')
                    exit(1)
            else:
                time.sleep(5)

    def showpage(self, nowPage, page):
        i = 0
        for i in range(0, len(nowPage)):
            if i < len(nowPage):
                oneStory = "\n\n" + nowPage[i].replace("\n\n", "").replace("<br/>", "\n") + "\n\n"
                print('第%d页,第%d个故事' % (page, i), oneStory)
                i += 1
            else:
                break

        myInput = str(input(u'回车键看下一页,按quit退出：\n'))
        if myInput == "quit":
            self.enable = False

    def start(self):
        self.enable = True
        page = self.page
        print('正在加载中请稍候......')
        # 新建一个线程在后台加载段子并存储
        _thread.start_new_thread(self.LoadPage, ())
        # ----------- 加载处理糗事百科 -----------
        while self.enable:
            # 如果self的page数组中存有元素
            if self.pages:
                nowPage = self.pages[0]
                del self.pages[0]
                self.showpage(nowPage, page)
                page += 1

if __name__ == "__main__":
    print(u'请按下回车浏览今日的糗百内容：')
    input(' ')
    myModel = SpiderModel()
    myModel.start()
