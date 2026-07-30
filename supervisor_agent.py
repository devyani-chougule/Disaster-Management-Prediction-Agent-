def supervisor_agent(disaster_type):


    if disaster_type == "Flood":

        return "flood_agent"


    elif disaster_type == "Cyclone":

        return "cyclone_agent"


    else:

        return "unknown"