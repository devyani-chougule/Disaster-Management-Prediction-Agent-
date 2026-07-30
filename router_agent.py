def detect_disaster(X):


    columns = set(X.columns)


    flood_features = {
        "precip_1d",
        "precip_3d",
        "NDVI",
        "NDWI",
        "elevation",
        "slope",
        "upstream_area",
        "TWI"
    }



    cyclone_features = {
        "Basin",
        "Longitude",
        "Latitude",
        "WindSpeed",
        "Pressure",
        "PressureDrop"
    }



    flood_match = len(
        columns.intersection(flood_features)
    )


    cyclone_match = len(
        columns.intersection(cyclone_features)
    )



    if flood_match > cyclone_match:

        return "Flood"



    elif cyclone_match > flood_match:

        return "Cyclone"



    else:

        return "Unknown"