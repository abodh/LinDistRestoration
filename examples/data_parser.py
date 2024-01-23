from ldrestoration.dssparser.dssparser import DSSManager        

def main():
    # dss_data = DSSManager(r"test_cases/ieee9500_dss/Master-unbal-initial-config.dss",
    #                       include_DERs=False,
    #                       include_secondary_network=False)
    # dss_data = DSSManager(r"test_cases/ieee123Bus/Run_IEEE123Bus.dss",
    #                       include_DERs=False,
    #                       include_secondary_network=False)
    
    dss_data = DSSManager(r"test_cases/ieee13Bus/IEEE13Nodeckt.dss",
                          include_DERs=False,
                          include_secondary_network=False)
    dss_data.parsedss()
    dss_data.saveparseddss(folder_name="parsed_data_13bus")

if __name__ == "__main__":
    main()
    
