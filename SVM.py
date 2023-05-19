import random


class image:
    pass

class svm:
    def divide(self):
        self.test = set()

        while len(self.test) < (4000 / 100 * 30):
            self.test.add(random.randint(1, 4000))
        
        # train = z(1, 4000) - test
    
    def readLabels(self):
        self.labels = []

        with open("labels.txt") as FILE:
            for line in FILE:  
                self.labels.append(int(line[0]))