class Hillslope_Link:


    def __init__(self):
        # an unique integer identifier for the link
        self.link_id=None
        #an vector of Hillslope Link objects
        self.upstream_link_id=None
        # a Hillslope Link object
        self.downstream_link_id=None
        #a vector of Hillslope_Link_Attribute object
        self.Hillslope_Link_Attribute=None 
    
    def __init__(self,link_id):
        self.link_id=link_id
    
    def set_link_id(self,link_id):
        if type(link_id) is int:
            self.link_id=link_id
    
    #modify, iterable
    def set_upstream_links(self, upstream_link_ids):
        if type(upstream_link_ids) is Hillslope_Link:
            self.upstream_link_id=upstream_link_ids
    
    def set_downstream_links(self,downstream_link_id):
        if type(downstream_link_id) is Hillslope_Link:
            self.downstream_link_id = downstream_link_id
    

