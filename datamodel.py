import pandas as pd


class DataModel:
    def __init__(self,df : pd.DataFrame,name : str) -> None:
        self._df = df
        self.name = name

    def new_df(self,new_df: pd.DataFrame):
        self._df = new_df

    
    def get_data(self):
        return self._df

    #Legacy DataModel Code
    def get_columns(self):
        if self._df.columns.isna().any():
            print("BAD READ OF COLUMNS")
            return
        else:
            return list(self._df.columns)
        
    def rename_columns(self):
        #Cleaning column names in DataFrame
        #I might move this to the parsing module
        try:
            if self._df.columns.isna().any():
                print("NaN IN COLUMN NAMES - HEADER SIZE ERROR")
                return
            else:
                cols = self.get_columns()
                for i in range(len(cols)):
                    #Changing column names to get rid of unwanted characters
                    col_name = cols[i]
                    col_name = col_name.replace(" ","_")
                    col_name = col_name.replace("-","_")
                    col_name = col_name.replace("(","")
                    col_name = col_name.replace("(","")
                    col_name = col_name.replace("/","_")
                    cols[i] = col_name
                self._df.columns = cols
                return
        except:
            print("BAD READ OF COLUMNS - HEADER SIZE ERROR")
            return