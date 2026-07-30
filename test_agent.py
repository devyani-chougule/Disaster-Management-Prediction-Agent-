import pandas as pd

from graph.disaster_graph import disaster_graph



# sample flood input

X = pd.DataFrame({

    "lon":[73.5],
    "lat":[18.5],
    "jrc_perm_water":[0.2],
    "precip_1d":[50],
    "precip_3d":[120],
    "NDVI":[0.4],
    "NDWI":[0.2],
    "landcover":[2],
    "elevation":[100],
    "slope":[5],
    "aspect":[10],
    "upstream_area":[20],
    "TWI":[8],
    "month":[7],
    "year":[2026],
    "day":[17]

})



result = disaster_graph.invoke(

{
    "X":X,
    "disaster_type":"",
    "prediction":{},
    "explanation":"",
    "guidelines":"",
    "report":""
}

)



print("Disaster Type:")
print(result["disaster_type"])


print("\nPrediction:")
print(result["prediction"])


print("\nSHAP:")
print(result["explanation"])


print("\nReport:")
print(result["report"])