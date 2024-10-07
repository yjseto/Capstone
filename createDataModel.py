import pandas as pd
import geopandas as gpd
from datamodel import DataModel
import re

def get_columns(df : pd.DataFrame):
        if df.columns.isna().any():
            print("BAD READ OF COLUMNS")
            return
        else:
            return list(df.columns)

def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    #Cleaning column names in DataFrame
    #I might move this to the parsing module
    try:
        if df.columns.isna().any():
            print("NaN IN COLUMN NAMES - HEADER SIZE ERROR")
            return
        else:
            cols = get_columns(df)
            for i in range(len(cols)):
                #Changing column names to get rid of unwanted characters
                col_name = cols[i]
                col_name = col_name.replace(" ","_")
                col_name = col_name.replace("-","_")
                col_name = col_name.replace("(","")
                col_name = col_name.replace("(","")
                col_name = col_name.replace("/","_")
                cols[i] = col_name
                df.columns = cols
            return df
    except:
        print("BAD READ OF COLUMNS - HEADER SIZE ERROR")
        return None
    
def load_shape_file(fp : str) -> gpd.GeoDataFrame:
    try:
        gdf = gpd.read_file(fp)
        print("SHAPE FILE FOUND")
        return gdf
    except:
        print("ERROR FETCHING FILE")
        return None

#has coords optional argument
def create_data_model(fp : str,header_size : int, has_coords : bool = None ) -> DataModel:
    in_data = pd.read_excel(fp,header = header_size) # will need to change the header size - likely to 0
    #need some error handling here
    #rename columns
    df = rename_columns(in_data)
    
    #pattern matching to replace bad values
    pattern = re.compile(r'''^(Null value|
                         Undetermined|
                         no Data|
                         Undetermined|
                         Not Specified|
                         Not Reported|
                         Missing|
                         )$''')

    df = df.map(lambda x: 'Unknown' if pattern.match(str(x)) else x)


    if has_coords:
        if 'Latitude' in df.columns and 'Longitude' in df.columns:
            print("Both 'Latitude' and 'Longitude' columns are present.")
            points_df = points_df = df[['Crash_Number','Longitude','Latitude']]
            points_gdf = gpd.GeoDataFrame(points_df, geometry= gpd.points_from_xy(points_df['Longitude'],points_df['Latitude']))

            #Load Shape files
            pop10Tiger  = load_shape_file("./data_processing/data/shp_files/TIGER_shp/tiger_blocks_ak.shp")
            urb_rural   = load_shape_file("./data_processing/data/shp_files/urb_rural_9_22/urb_rural.shp")

            points_into_population = gpd.sjoin(points_gdf,pop10Tiger,how="left",predicate="intersects")
            points_into_area = gpd.sjoin(points_gdf,urb_rural,how="left",predicate="intersects")
            points_into_area['type'] = points_into_area['type'].fillna('rural') #filling any values outside urban/suburban into rural category

            coords_pop10 = points_into_population[['Crash_Number','pop10']]
            
            coords_area = points_into_area[['Crash_Number','type']]
            coords_area = coords_area.rename(columns = {'type':'area'})

            df = pd.merge(df,coords_area,how='left',on='Crash_Number')
            df = pd.merge(df,coords_pop10,how='left',on='Crash_Number')
            #theoretically the df is now merged
            #and we now have crashes classified into urban and rural and have a population associated with it?
            
            #dropping lat and long as they are no longer needed?
            exclude_columns = ['Latitude','Longitude']
            df = df.drop(columns=exclude_columns)

        else:
            print("One or both columns are missing.")
    
    df = df.drop_duplicates(subset='Crash_Number')
    df = df.reset_index(drop=True)
    dm = DataModel(df,"testData")

    return dm
