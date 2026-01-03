
import pandas as pd
import math

from datetime import datetime
from loaders import TelemetryLoader
from collections import Counter

telemetry_loader = TelemetryLoader(2025, 'Silverstone', 'R')

session = telemetry_loader.get_session()
session.load()
laps = session.laps
ant_laps = session.laps.pick_drivers(['ANT'])
ant_laps.to_csv('ant_laps.csv')
laps.reset_index(drop=True)
best_sector_1_time = min(ant_laps['Sector1Time'].dropna())
print(best_sector_1_time)
# lap_start_times = ant_laps['LapStartDate']
# print(lap_start_times)
# print(lap_start_times.iloc[0].to_pydatetime())
# def get_lap(time, lap_start_times):
#     replay_time = datetime.strptime(time + '000', '%Y-%m-%d %H-%M-%S-%f')
    
