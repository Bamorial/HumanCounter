class Point:
    def __init__(self, x, y):
        self.x=x
        self.y=y
    def __repr__(self):
        return "x: "+str(self.x)+ ", y: "+str(self.y)