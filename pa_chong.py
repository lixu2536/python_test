# -*- coding: utf-8 -*-
import requests
import time

headers = {
    "cookie": "appmsglist_action_3889613222=card; ua_id=Q1Dfu2THA6T9Qr1HAAAAAN_KYa5xTwNmiuqj1Mkl6PY=; wxuin=18828715020059xid=a5c7612f529374b74deb4178e7ff4ca7",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36"
}
url = 'https://mp.weixin.qq.com/cgi-bin/appmsg'
fad = 'MjM5ODM3MTUwMA=='                     #爬不同公众号只需要更改 fakeid

def page(num=1):                             #要请求的文章页数
    title = []
    link = []
    for i in range(num):
        data = {
            'action': 'list_ex',
            'begin': i*5,       #页数
            'count': '5',
            'fakeid': fad,
            'type': '9',
            'query':'' ,
            'token': '1753262244',
            'lang': 'zh_CN',
            'f': 'json',
            'ajax': '1',
        }
        r = requests.get(url,headers = headers,params=data)
        dic = r.json()
        for i in dic['app_msg_list']:     # 遍历dic['app_msg_list']中所有内容
            title.append(i['title'])      # 取 key键 为‘title’的 value值
            link.append(i['link'])        # 去 key键 为‘link’的 value值
    return title,link

if __name__ == '__main__':
    (tle,lik) = page(5)
    for x,y in zip(tle,lik):
        print(x,y)