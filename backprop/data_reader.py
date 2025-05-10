class DataReader:
    
    def __init__(self, path):
        self.path = path
        
    def read_csv_file(self):
        """Read csv file (without using any libraries)"""
        feature = []
        label = []
        count = 0
        path = self.path
        with open(path) as f:
            lines = f.readlines()
            for line in lines:
                if count == 0:
                    count += 1
                    continue 
                
                xy = line.splitlines()
                print(xy)
                x1, x2, y = xy[0].split(",")
                print(x1)
                print(x2)
                print(y)
                feature.append([float(x1), float(x2)])  
                label.append(float(y))    
                
        return feature, label  

