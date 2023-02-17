class HLM:
    name='hlm'
    #dict links
    #key is an integer
    #value is the link object
    dict_links={}
    def __init__(self) -> None:
        pass

    def add_link(self,link):
        self.dict_links[link.id]=link

    def set_link_att(self,id,key,val):
        self.dict_links[id].set_att(key,val)
    

class Link():
    name='link'
    #dict_att={}
    def __init__(self,id:int):
        self.id=id
        self.dict_att={}
    def set_att(self,key:str,val:float):
        self.dict_att[key]=val
        

def model1(link: Link)->None:
    pass

def model2(link:Link):
#https://stackoverflow.com/questions/16909779/any-way-to-solve-a-system-of-coupled-differential-equations-in-python
    pass
#test1
def _test1():
    hlm = HLM()
    link1 = Link()
    link1.dict_att['key1']='value1'
    link1.dict_att[2]=-99
    link1.dict_att['area']=100
    link1.id=1
    hlm.dict_Links[link1.id]=link1

#test 2
def test2():
    link1 = Link(1)
    link1.set_att('area',20)
#print(help(Link))
def test3():
    hlm = HLM()

    link1 = Link(1)
    hlm.add_link(link1)
    hlm.dict_links[1].id
    print(help(link1))

def test4():
    #run a model that passes 20% of water to link downstream
    
    def model2(link:Link):
        link.dict_att['out'] = link.dict_att['init']*0.2
        link.dict_att['init']*=0.8
    
    def model3(linkup:Link,linkdown:Link):
        linkdown.dict_att['init']+=linkup.dict_att['out']
    
    linkup = Link(1)
    linkup.dict_att['init']=10
    model2(linkup)
    linkdown = Link(1)
    linkdown.dict_att['init']=0
    model3(linkup,linkdown)
    

