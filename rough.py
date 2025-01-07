from meteostat import Point, Daily, Hourly

import pandas as pd

#Hourly Data

start = pd.to_datetime("2023-1-1")
end = pd.to_datetime("2024-12-15 23:59")
vancouver = Point(23.0225,72.5714)
hourly_data = Hourly(vancouver,start,end)
hourly_data = hourly_data.fetch()
hourly_data.reset_index(inplace=True)
print(hourly_data.head())