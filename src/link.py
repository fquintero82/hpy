from multiprocessing.spawn import _main
from pip import main


class Link:


    def __init__(self):
        # an unique integer identifier for the link
        self.id=None
        # a vector of Hillslope Link objects
        self.upstream_links=[]
        self.num_upstream_links=None
        # a Hillslope Link object
        self.downstream_link_id=None
        #a vector of Hillslope_Link_Attribute object
        #self.Hillslope_Link_Attribute=None 
        # link length in meters
        self.len_meters=None
        self.drainage_area_meters =None
        
    
    def __init__(self,id):
        self.id=id
    
    def set_id(self,id):
        if type(id) is int:
            self.id=id
    
    def get_id(self):
        return self.id

    #modify, iterable
    def set_upstream_links(self, upstream_links):
        self.num_upstream_links = len(upstream_links)
        for x in upstream_links:
            if type(x) is Link:
                self.upstream_links.add(x)
    
    def get_upstream_links(self):
        return(self.upstream_links)

    def set_downstream_links(self,downstream_link_id):
        if type(downstream_link_id) is Link:
            self.downstream_link_id = downstream_link_id
    
    def get_len(self):
        return(self.len_meters)

    def set_drainage_area(self,myDrainageArea):
        if type(myDrainageArea) is int:
            self.drainage_area_meters=myDrainageArea
    
    def get_drainage_area(self):
        return(self.drainage_area_meters)
    
    def get_num_upstream_links(self):
        return(self.num_upstream_links)


