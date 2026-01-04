
import fastf1
import numpy as np
import os

class TelemetryLoader:
    def __init__(self, year, gp, identifier, backend='fastf1', cache_dir='./.fastf1_cache/'):
        self.year = year
        self.gp = gp
        self.identifier = identifier
        self.backend = backend
        self.cache_dir = cache_dir
    
    def get_session(self):
        fastf1.Cache.enable_cache(self.cache_dir)
        session = fastf1.get_session(
            year=self.year,
            gp=self.gp,
            identifier=self.identifier,
            backend=self.backend
        )
        session.load()
        return session
    
class RacetrackDatabaseLoader:
    def __init__(self, gp, database_dir='./racetrack-database/'):
        self.gp = gp
        self.database_dir = database_dir
    
    def get_track_data(self):
        return np.genfromtxt(os.path.join(self.database_dir, 'tracks/', f"{self.gp}.csv"), delimiter=',', names=True, comments='#', autostrip=True)
    
    def get_raceline_data(self):
        return np.genfromtxt(os.path.join(self.database_dir, 'reacelines/', f"{self.gp}.csv"), delimiter=',', names=True, comments='#', autostrip=True)