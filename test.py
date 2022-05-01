from sklearn.ensemble import RandomForestClassifier
import numpy as np

domainlist = []


def feature(s):
    l = len(s)
    n = 0
    for c in s:
        if 48 <= ord(c) < 58:
            n += 1
    _, counts = np.unique(list(s), return_counts=True)
    total = sum(counts)
    percent = list(map(lambda x: x / total, counts))
    return [l, n, sum(-n * np.log(n) for n in percent)]


class Domain:
    def __init__(self, _name, _label):
        self.name = _name
        self.label = _label

    def returnData(self):
        return feature(self.name)

    def returnLabel(self):
        if self.label == "notdga":
            return 0
        else:
            return 1


def initData(filename):
    with open(filename) as f:
        for line in f:
            line = line.strip()
            tokens = line.split(",")
            name = tokens[0]
            label = tokens[1]
            domainlist.append(Domain(name, label))


def main():
    print("Initialize Raw Objects")
    initData("train.txt")
    featureMatrix = []
    labelList = []
    print("Initialize Matrix")
    for item in domainlist:
        featureMatrix.append(item.returnData())
        labelList.append(item.returnLabel())
    print(featureMatrix)
    print("Begin Training")
    clf = RandomForestClassifier(random_state=0)
    clf.fit(featureMatrix, labelList)
    print("Begin Predicting")

    f = open("result.txt", "w+")
    test = open("test.txt")
    line = test.readline()
    while line:
        if clf.predict([feature(line)]) == [0]:
            s = "notdga"
        else:
            s = "dga"
        result = line.strip() + "," + s + "\n"
        f.write(result)
        line = test.readline()

    f.close()
    test.close()


if __name__ == '__main__':
    main()