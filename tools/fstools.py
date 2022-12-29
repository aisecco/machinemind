import os


def makedirs(forder):
    if not os.path.exists(forder):
        os.makedirs(forder)
        # os.mkdir(forder)


def getfiles(forder):
    for root, dirs, files in os.walk(forder):
        return files

def getsubdirs(forder):
    for root, dirs, files in os.walk(forder):
        return dirs

def getpath(forder, file):
    return os.path.join(forder, file)


def getfolders(forder):
    for root, dirs, files in os.walk(forder):
        return dirs


def getInfo(forder):
    files = getfiles(forder)
    print("list files: ", forder)
    for f in files:
        print(f)

    # 遍历所有的文件夹
    dirs = getfolders(forder)
    print("list folers: ", forder)
    for d in dirs:
        print(d)


def main():
    getInfo("/Users/mac/dev/tensorflow2/dataset/KDD99/split/")


if __name__ == '__main__':
    main()
