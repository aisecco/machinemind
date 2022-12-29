#!/usr/bin/python

import fileinput
import re


def scr():
    all1 = ''
    for line in fileinput.input("../data/result.log"):
        all1 = all1 + line
        # print line,
    return all1
n = 1
def extr(text):
    pattern = r'\s.*wrk.*\-t(\d*) -c(\d*) -d(\d*).*\sRunning (.*) test @ (.*)'
    # pattern = r'(.*)latency (.*?) '
    seo = re.search(pattern, text, re.M | re.I)
    if seo:

        print(seo.group())
        n = 1
        while n <= seo.lastindex:
            print(n, seo.group(n))
            n = n + 1


def extract(text, pattern):
    seo = re.search(pattern, text, re.M | re.I)
    if seo:
        print(seo.group())
        n = 1
        while n <= seo.lastindex:
            print(n, seo.group(n))
            n = n + 1


if __name__ == "__main__":
    text = scr()

    ps = []
    pattern = r'wrk.*\-t(\d*) -c(\d*) -d(\d*).*'
    ps.append(pattern)

    pattern = r'Running (.*) test @ (.*)'
    ps.append(pattern)

    pattern = r'Latency[ ]+(\S+)[ ]+(\S+)[ ]+(\S+)[ ]+(.+\%)'
    ps.append(pattern)

    pattern = r'Req/Sec\s+(\S+)\s+(\S+)\s+(\S+)\s+(.+\%)'
    ps.append(pattern)

    pattern = r'Latency Distribution\s+50\%\s+(\S+)\s+75\%\s+(\S+)\s+90\%\s+(\S+)\s+99%\s+(\S+)'
    ps.append(pattern)

    pattern = r'(\S+)\s+requests in\s+(.*),\s+(.*) read'
    ps.append(pattern)

    pattern = r'Socket errors: connect\s+(.*), read\s+(.*), write\s+(.*), timeout\s+(.*)'
    ps.append(pattern)

    pattern = r'Requests/sec:\s+(.*)\s+Transfer/sec:\s+(.*)'
    ps.append(pattern)

    for p in ps:
        extract(text, p)
    # extr(text)
