from createDataModel import create_data_model

def main():
    new_dm = create_data_model("./data_processing/data/testData_2.xlsx", header_size=0, has_coords=True)
    print(new_dm.get_data())

if __name__ == "__main__":
    main()