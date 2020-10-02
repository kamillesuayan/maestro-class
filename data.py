class Data:
    '''
    '''
    def __init__(self, data, access) -> None:
        self.data = data
        self.access = access
    def read(self,idx):
        return self.data[idx]
    def write(self):
        pass