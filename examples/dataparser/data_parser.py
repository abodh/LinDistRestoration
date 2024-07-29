from ldrestoration import DSSManager


def main():
    dss_data = DSSManager(
        r"../test_cases/ieee9500_dss/Master-unbal-initial-config.dss",
        include_DERs=False,
        include_secondary_network=False,
    )
    dss_data.parsedss()
    dss_data.saveparseddss(folder_name="parsed_data_9500_noder", folder_exist_ok=True)


if __name__ == "__main__":
    main()
