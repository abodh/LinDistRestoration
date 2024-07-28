from ldrestoration import DSSManager        

def main():
    dss_data = DSSManager(r"../test_cases/ieee123Bus/Run_IEEE123Bus.dss",
                          include_DERs=False,
                          include_secondary_network=False)
    dss_data.parsedss()
    dss_data.saveparseddss(folder_name="parsed_data_ieee123", folder_exist_ok=True)

if __name__ == "__main__":
    main()
    
