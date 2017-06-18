'''
filename: image_resize.py

Created on April 17, 2017

@author: Zelong Cong , University of Melbourne

This application is used to download a massive nubmer of images through links given by ImageNet

The URLs need to be stored in a txt file before using this application.
'''

import urllib.request
import socket
import threading
import queue

import os

myqueue=queue.Queue()

socket.setdefaulttimeout(5.0)
global count
count =0

lock=threading.Lock()

# read file that contains urls,
def read_file():
    with open('/Users/nick/Dropbox/Python/PythonWorkplace/Tools/worker_url.txt', 'r') as reader:
        for each in reader.readlines():
            myqueue.put(each)
    return myqueue
# dowload and name it:

def download(url):
    try:
        global count

        urllib.request.urlretrieve(url,'/Users/nick/Dropbox/Python/PythonWorkplace/Tools/goggle_imageNet/pic_'+'%s.jpg' % count)
    except Exception:
        print("error",Exception)
        return 0
    print("pic "+count.__str__()+" finished")
    print("downloaded "+url)
    lock.acquire(True)
    try :
        count=count+1
    finally:
        lock.release()


def run():
    while myqueue.not_empty:
        download(myqueue.get())


def main():
    myqueue=read_file()

    if not os.path.isdir('/Users/nick/Dropbox/Python/PythonWorkplace/Tools/goggle_imageNet'):
        os.mkdir('/Users/nick/Dropbox/Python/PythonWorkplace/Tools/goggle_imageNet')

    print("finish reading the file")

    for each in range(8):
        threading.Thread(target=run).start()


if __name__ == '__main__':
    main()