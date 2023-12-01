import pandas as pd

class dataset:
    row = 0
    column = 0
    
    def __init__(self, file_path):
        # dataset_path = input("请输入数据集的绝对路径：")
        self.path = file_path
        # self.name = (self.path.split("\\"))[-2]
        self.name = 'flights'
        self.readCSV()
        self.df_to_dictionary()

    def readCSV(self):
        self.df = pd.read_csv(self.path).astype('str')
        self.df.insert(0, 'index', list(range(len(self.df))))
        self.df.fillna('nan',inplace=True)
        self.row = self.df.shape[0]
        self.column = self.df.shape[1]
        return self.df

    def df_to_dictionary(self):
        self.dict = {}
        for i in range(self.df.shape[0]):
            self.dict[i] = self.df.loc[i]
        return self.dict
    
    def read_clean(self, file_path):
        self.clean_df = pd.read_csv(file_path).astype('str')
        self.clean_df.insert(0, 'index', list(range(len(self.clean_df))))
        self.clean_df.fillna('nan',inplace=True)
        return self.clean_df


#  E:\Graduation Project\Daisy\datasets\hospital\dirty.csv
if __name__ == '__main__':
    dt = dataset()
    dt.readCSV()
    dt.df_to_dictionary()
    print(dt.dict[1]["index"])
