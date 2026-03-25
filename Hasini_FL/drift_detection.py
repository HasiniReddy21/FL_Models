class CUSUM:

    def __init__(self,threshold=5):

        self.threshold = threshold
        self.sum = 0

    def detect(self,error):

        self.sum = max(0,self.sum + error)

        if self.sum > self.threshold:

            print("Concept Drift Detected")

            self.sum = 0

            return True

        return False