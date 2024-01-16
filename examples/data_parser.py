from ldrestoration.dssparser.dssparser import DSSManager        

def main():
    dss_data = DSSManager(r"test_cases/ieee9500_dss/Master-unbal-initial-config.dss",
                          include_DERs=True,
                          include_secondary_network=False)
    dss_data.parsedss()
    dss_data.saveparseddss()

if __name__ == "__main__":
    main()
    
