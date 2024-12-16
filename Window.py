from Point import Point
class Window:
    def __init__(self, corner: Point, width: int, height: int):
        self.corner=corner
        self.width=width
        self.height=height
    def __repr__(self):
        return "corner: "+ str(self.corner)+', width: '+ str(self.width)+', height: '+ str(self.height)