import pyarrow as pa

BUS_DATA = pa.schema(
    [
        ("name", pa.string()),  # unique name of the bus
        ("base_kv", pa.float64()),  # base voltage of the bus in kV
        ("latitude", pa.float64()),  # latitude of the bus location
        ("longitude", pa.float64()),  # longitude of the bus location
    ]
)

TRANSFORMER_DATA = pa.schema(
    [
        ("name", pa.string()),  # unique name of the transformer
        ("num_windings", pa.int64()),  # number of windings of the transformer
        ("num_phases", pa.int64()),  # number of phases of the transformer
        ("connected_from", pa.string()),  # name of the bus the transformer is connected from
        ("connected_to", pa.string()),  # name of the bus the transformer is connected to
        ("kva_rating", pa.float64()),  # kva rating of the transformer
        ("primary_kv", pa.float64()),  # primary voltage of the transformer in kV
        ("secondary_kv", pa.float64()),  # secondary voltage of the transformer in kV
        ("primary_connection", pa.string()),  # primary connection type of the transformer (wye or delta)
        ("secondary_connection", pa.string()),  # secondary connection type of the transformer (wye or delta)
        (
            "tertiary_connection",
            pa.string(),
        ),  # tertiary connection type of the transformer (wye or delta or "n/a" if no tertiary winding)
    ]
)
