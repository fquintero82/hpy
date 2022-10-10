class Hillslope:
    def __init__(self):
        #unique Link
        self.id=None
        self.area_meters = None
        

    def set_id(self,id):
        if type(id) is int:
            self.id = id

    def get_id(self):
        return self.id

    def set_area(self,myArea):
        if type(myArea) is float:
            self.area_meters=myArea

    def get_area(self):
        return(self.area_meters)



        