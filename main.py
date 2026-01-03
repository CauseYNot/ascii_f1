
import math
import time
import shutil
import pandas as pd

from datetime import datetime
from itertools import groupby
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live

from loaders import *

DRS_KEY = {
    0: '[bold red]OFF[/]',
    1: '[bold red]OFF[/]',
    8: '[bold orange]DETECTED ELIGIBLE[/]',
    10: '[bold green]ON[/]',
    12: '[bold green]ON[/]',
    14: '[bold green]ON[/]',
}

TYRE_KEY = {
    'SOFT': '#E10600',
    'MEDIUM': '#FFD100',
    'HARD': '#FFFFFF',
    'INTERMEDIATE': '#00A651',
    'WET': '#00AEEF'
}

CONSTRUCTOR_COLOUR_KEY = {
    'Mercedes': '#00D7B6',
    'Red Bull Racing': '#4781D7',
    'Ferrari': '#ED1131',
    'McLaren': '#F47600',
    'Alpine': '#00A1E8',
    'Racing Bulls': '#6C98FF',
    'Aston Martin': '#229971',
    'Williams': '#1868DB',
    'Kick Sauber': '#01C00E',
    'Haas F1 Team': '#9C9FA2',
}
def _generate_throttle_gradient(n_colors):
    s = (255, 0, 0) # FF0000 (red)
    f = (0, 255, 0) # 00FF00 (green)
    gradient_hex = []
    for i in range(n_colors):
        # linear interpolation (lerp)
        t = i / (n_colors - 1) if n_colors > 1 else 0
        r = int(s[0] + (f[0] - s[0]) * t)
        g = int(s[1] + (f[1] - s[1]) * t)
        b = int(s[2] + (f[2] - s[2]) * t)
        gradient_hex.append(f"#{r:02X}{g:02X}{b:02X}")
        
    return gradient_hex

class DriverViewAsciiPanel:
    def __init__(self, panel_width, panel_height, telemetry, track_data, fov, lookahead, camera_height, horizon_y, vertical_scale):
        self.telemetry = telemetry
        self.track_data = track_data
        self.panel_width = panel_width
        self.panel_height = panel_height
        self.fov = fov
        self.lookahead = lookahead
        self.camera_height = camera_height
        self.horizon_y = horizon_y
        self.vertical_scale = vertical_scale

        self.x_car = self.telemetry['X'].to_numpy()
        self.y_car = self.telemetry['Y'].to_numpy()
        self.x_track = self.track_data['x_m']
        self.y_track = self.track_data['y_m']
        self.width_left = self.track_data['w_tr_left_m']
        self.width_right = self.track_data['w_tr_right_m']
        
        self.x_car_min, self.x_car_max = self.x_car.min(), self.x_car.max()
        self.y_car_min, self.y_car_max = self.y_car.min(), self.y_car.max()
        self.x_track_min, self.x_track_max = self.x_track.min(), self.x_track.max()
        self.y_track_min, self.y_track_max = self.y_track.min(), self.y_track.max()

    def _calculate_heading(self, i, window=1):
        i0 = max(0, i - window)
        i1 = min(len(self.x_car) - 1, i + window)

        dx = self.x_car[i1] - self.x_car[i0]
        dy = self.y_car[i1] - self.y_car[i0]

        return math.atan2(dy, dx) - math.pi/2

    def _car_to_track_co_ords(self, car_x, car_y):
        x_car_fraction = (car_x - self.x_car_min) / (self.x_car_max - self.x_car_min)
        x_track_converted = self.x_track_min + (x_car_fraction) * (self.x_track_max - self.x_track_min)
        y_car_fraction = (car_y - self.y_car_min) / (self.y_car_max - self.y_car_min)
        y_track_converted = self.y_track_min + (y_car_fraction) * (self.y_track_max - self.y_track_min)

        return x_track_converted, y_track_converted

    def _project_point(self, point_x, point_y, cam_x, cam_y, heading):
        dx = point_x - cam_x
        dy = point_y - cam_y

        cos_h = math.cos(-heading)
        sin_h = math.sin(-heading)

        # camera-space coordinates
        x_rel = dx * cos_h - dy * sin_h   # left/right
        y_rel = dx * sin_h + dy * cos_h   # forward

        if y_rel <= 0:
            return None

        # perspective projection 
        inv_z = 1.0 / y_rel
        screen_x = int(self.panel_width // 2 + x_rel * inv_z * self.fov)
        # vertical projection using camera height
        screen_y = int(
            self.horizon_y * self.panel_height
            + self.camera_height * inv_z * self.panel_height * self.vertical_scale
        )

        if 0 <= screen_y < self.panel_height and 0 <= screen_x < self.panel_width:
            return screen_y, screen_x
        
        return None

    def _build_track_points(self, i):
        left_points = []
        centre_points = []
        right_points = []

        cam_heading = self._calculate_heading(i, 1)
        cam_x, cam_y = self._car_to_track_co_ords(self.x_car[i], self.y_car[i])

        n = len(self.x_track)

        for j in range(n):
            xt = self.x_track[j]
            yt = self.y_track[j]
            wl = self.width_left[j]
            wr = self.width_right[j]

            # estimate track tangent using forward difference
            if j < n - 1:
                dx = self.x_track[j + 1] - xt
                dy = self.y_track[j + 1] - yt
            else:
                # default to the backward difference if we're at the end of the lap
                dx = xt - self.x_track[j - 1]
                dy = yt - self.y_track[j - 1]

            track_heading = math.atan2(dy, dx)

            normal_x = -math.sin(track_heading)
            normal_y =  math.cos(track_heading)

            x_left, y_left = xt + normal_x * wl, yt + normal_y * wl
            x_right, y_right = xt - normal_x * wr, yt - normal_y * wr

            left_point = self._project_point(x_left, y_left, cam_x, cam_y, cam_heading)
            centre_point = self._project_point(xt, yt, cam_x, cam_y, cam_heading)
            right_point = self._project_point(x_right, y_right, cam_x, cam_y, cam_heading)

            left_points.append(left_point)
            centre_points.append(centre_point)
            right_points.append(right_point)

        return left_points, centre_points, right_points

    def generate_frame(self, i):
        left_points, centre_points, right_points = self._build_track_points(i)
        buf = [[' '] * self.panel_width for _ in range(self.panel_height)]
        for left_point, centre_point, right_point in zip(left_points, centre_points, right_points):
            if left_point is not None:
                row, col = left_point
                if 0 <= row < self.panel_height and 0 <= col < self.panel_width:
                    buf[row][col] = '|'
                    
            if centre_point is not None:
                row, col = centre_point
                if 0 <= row < self.panel_height and 0 <= col < self.panel_width:
                    buf[row][col] = '.'
            
            if right_point is not None:
                row, col = right_point
                if 0 <= row < self.panel_height and 0 <= col < self.panel_width:
                    buf[row][col] = '|'
        
                    
        return '\n'.join(''.join(row) for row in buf)



class TelemetryAsciiPanel:
    def __init__(self, panel_width, panel_height, telemetry):
        self.panel_width = panel_width
        self.panel_height = panel_height
        self.telemetry = telemetry

    def generate_frame(self, i):
        bar_colours = _generate_throttle_gradient(self.panel_width // 2)
        
        speed = self.telemetry['Speed'].iloc[i]
        rpm = int(self.telemetry['RPM'].iloc[i])
        throttle_percent = int(self.telemetry['Throttle'].iloc[i])
        throttle_bar = ''.join(f'[{bar_colours[idx]}]█[/]' if idx <= throttle_percent // (100/len(bar_colours)) else ' ' for idx in range(len(bar_colours)))
        brake = "[bold green]ON[/]" if self.telemetry['Brake'].iloc[i] else "[bold red]OFF[/]"
        gear = self.telemetry['nGear'].iloc[i]
        gear = f'[bold #00d7ff]{gear}[/]' if gear != 0 else '[bold #00d7ff]N[/]'
        drs = DRS_KEY[self.telemetry['DRS'].iloc[i]]
        date = self.telemetry['Date'].iloc[i]
        session_time = self.telemetry['SessionTime'].iloc[i]
        
        return f"""Speed: {int(speed)} km/h
RPM: {rpm}
Throttle: |{throttle_bar}| {throttle_percent}%
Brake: {brake}
Gear: {gear}
DRS: {drs}
Session Time: {session_time}
Date: {date}"""


class LapDataPanel:
    def __init__(self, panel_width, panel_height, laps_data):
        self.panel_width = panel_width
        self.panel_height = panel_height
        self.laps_data = laps_data
    
    def _tyre_strategy_diagram(self, lap):
        compounds = list(laps_data['Compound'])[:lap]
        stints = [[compound, len(list(group))] for compound, group in groupby(compounds)]
        shrink_factor = (self.panel_width - 30) / len(laps_data)
        return '|'.join(f"[{TYRE_KEY[stint[0]]}]{"█"*int(stint[1]*shrink_factor)} {stint[1]} [/]" for stint in stints)
    
    def generate_frame(self, lap):
        lap_data = self.laps_data.iloc[lap]
        return f"""[bold underline]Driver:[/] [{CONSTRUCTOR_COLOUR_KEY[lap_data['Team']]}]{lap_data['DriverNumber']} - {lap_data['Driver']} ({lap_data['Team']})[/]
[bold underline]Lap: {lap}[/]
Tyre Compound: [{TYRE_KEY[lap_data['Compound']]}]{lap_data['Compound']}[/]
Tyre Age: {int(lap_data['TyreLife'])}
Stint: {int(lap_data['Stint'])}
Position: {int(lap_data['Position'])}
Tyre Strategy: |{self._tyre_strategy_diagram(lap=lap)}"""


class SectorTimingPanel:
    def __init__(self, panel_width, panel_height, laps_data):
        self.panel_width = panel_width
        self.panel_height = panel_height
        self.laps_data = laps_data
        self.telemetry = laps_data.telemetry
    
    def _tyre_strategy_diagram(self, lap):
        compounds = list(laps_data['Compound'])[:lap]
        stints = [[compound, len(list(group))] for compound, group in groupby(compounds)]
        shrink_factor = (self.panel_width - 30) / len(laps_data)
        return '|'.join(f"[{TYRE_KEY[stint[0]]}]{"█"*max(1, int(stint[1]*shrink_factor))} {stint[1]} [/]" for stint in stints)
    
    def generate_frame(self, lap, i):
        current_telemetry_data = self.telemetry.iloc[i]
        lap_data = self.laps_data.iloc[lap - 1]
        lap_start_time = lap_data['LapStartTime']
        best_sector_1_time = min(laps_data['Sector1Time'].dropna().iloc[:3])
        best_sector_2_time = min(laps_data['Sector2Time'].dropna().iloc[:3])
        best_sector_3_time = min(laps_data['Sector3Time'].dropna().iloc[:3])
        current_session_time = current_telemetry_data['SessionTime']
        sector_1_session_time = lap_data['Sector1SessionTime']
        sector_2_session_time = lap_data['Sector2SessionTime']
        sector_3_session_time = lap_data['Sector3SessionTime']
        sector_1_time = (sector_1_session_time - lap_start_time) if not pd.isna(sector_1_session_time) else pd.Timedelta(0.0, 's')
        sector_2_time = (sector_2_session_time - lap_start_time - sector_1_time) if not pd.isna(sector_2_session_time) else pd.Timedelta(0.0, 's')
        sector_3_time = (sector_3_session_time - lap_start_time - sector_1_time - sector_2_time) if not pd.isna(sector_3_session_time) else pd.Timedelta(0.0, 's')

        if current_session_time < sector_1_session_time:
            sector_1_time, sector_2_time, sector_3_time = current_session_time - lap_start_time, '', ''
        elif current_session_time < sector_2_session_time:
            sector_2_time, sector_3_time = current_session_time - sector_1_session_time, ''
        elif current_session_time < sector_3_session_time:
            sector_3_time = current_session_time - sector_2_session_time
        
        return f"""
S1: {sector_1_session_time}, S2: {sector_2_session_time}, S3: {sector_3_session_time} Current {current_session_time}
S1: {sector_1_time}, S2: {sector_2_time}, S3: {sector_3_time}"""


class MinimapAsciiPanel:
    def __init__(self, panel_width, panel_height, laps_data, track_data):
        self.panel_width = panel_width
        self.panel_height = panel_height
        self.laps_data = laps_data
        self.telemetry = laps_data.telemetry
        self.track_data = track_data

        self.x_car = self.telemetry['X'].to_numpy()
        self.y_car = self.telemetry['Y'].to_numpy()
        self.x_track = self.track_data['x_m']
        self.y_track = self.track_data['y_m']
        
        self.x_car_min, self.x_car_max = self.x_car.min(), self.x_car.max()
        self.y_car_min, self.y_car_max = self.y_car.min(), self.y_car.max()
        self.x_track_min, self.x_track_max = self.x_track.min(), self.x_track.max()
        self.y_track_min, self.y_track_max = self.y_track.min(), self.y_track.max()

    def _tel_to_screen(self, x, y):
        gx = int((x - self.x_car_min) / (self.x_car_max - self.x_car_min) * (self.panel_width - 5))
        gy = self.panel_height - 1 - int((y - self.y_car_min) / (self.y_car_max - self.y_car_min) * (self.panel_height - 10)) - 5 
        return gx, gy

    def _track_to_screen(self, x, y):
        gx = int((x - self.x_track_min) / (self.x_track_max - self.x_track_min) * (self.panel_width - 5))
        gy = self.panel_height - 1 - int((y - self.y_track_min) / (self.y_track_max - self.y_track_min) * (self.panel_height - 10)) - 5
        return gx, gy

    def generate_frame(self, i):
        buf = [[" "] * self.panel_width for _ in range(self.panel_height)]
        for k in range(len(self.x_track)):
            cx, cy = self._track_to_screen(self.x_track[k], self.y_track[k])
            if 0 <= cx < self.panel_width and 0 <= cy < self.panel_height:
                buf[cy][cx] = "#"

        gx, gy = self._tel_to_screen(self.x_car[i], self.y_car[i])
        if 0 <= gx < self.panel_width and 0 <= gy < self.panel_height:
            buf[gy][gx] = f"[{CONSTRUCTOR_COLOUR_KEY[self.laps_data['Team'].iloc[0]]}]○[/]"
        
        return "\n".join("".join(row) for row in buf)



class F1AsciiReplayDisplay:
    def __init__(self, laps_data, track_data, terminal_width=None, terminal_height=None, fov=40.0, lookahead=50.0, camera_height=0.5, horizon_y=0.1, vertical_scale=1.2, refresh_rate=1/60):
        self.terminal_width = terminal_width or (shutil.get_terminal_size()[0])
        self.terminal_height = terminal_height or (shutil.get_terminal_size()[1])
        
        self.laps_data = laps_data
        self.telemetry = laps_data.telemetry
        self.refresh_rate = refresh_rate
        
        self.driver_view_panel_width = int((self.terminal_width - 2) * 0.65)
        self.driver_view_panel_height = self.terminal_height - 2
        
        self.minimap_panel_width = self.terminal_width - self.driver_view_panel_width
        self.minimap_panel_height = self.terminal_height // 2 - 2
        
        self.telemetry_panel_width = self.minimap_panel_width
        self.telemetry_panel_height = 8
        
        self.lap_data_panel_width = self.minimap_panel_width // 2
        self.lap_data_panel_height = self.terminal_height - self.telemetry_panel_height - self.minimap_panel_height - 6
        
        self.sector_timing_panel_width = self.lap_data_panel_width
        self.sector_timing_panel_height = self.lap_data_panel_height
        
        # self.telemetry_panel_width = self.minimap_panel_width
        # self.telemetry_panel_height = self.terminal_height - self.driver_view_panel_height - 4
        
        # self.lap_data_panel_width = self.minimap_panel_width
        # self.lap_data_panel_height = self.terminal_height - self.telemetry_panel_height - self.driver_view_panel_height - 6
        
        self.driver_view_panel = DriverViewAsciiPanel(
            panel_width=self.driver_view_panel_width,
            panel_height=self.driver_view_panel_height,
            telemetry=self.telemetry,
            track_data=track_data,
            fov=fov,
            lookahead=lookahead,
            camera_height=camera_height,
            horizon_y=horizon_y,
            vertical_scale=vertical_scale
        )
        
        self.telemetry_panel = TelemetryAsciiPanel(
            panel_width=self.telemetry_panel_width,
            panel_height=self.telemetry_panel_height,
            telemetry=self.telemetry
        )
        
        self.lap_data_panel = LapDataPanel(
            panel_width=self.lap_data_panel_width,
            panel_height=self.lap_data_panel_height,
            laps_data=laps_data,
        )
        
        self.sector_timing_panel = SectorTimingPanel(
            panel_width=self.lap_data_panel_width,
            panel_height=self.lap_data_panel_height,
            laps_data=laps_data
        )
        
        self.minimap_panel = MinimapAsciiPanel(
            panel_width=self.minimap_panel_width,
            panel_height=self.minimap_panel_height,
            laps_data=laps_data,
            track_data=track_data
        )
    
    def main(self):
        layout = Layout()
        layout.split_row(
            Layout(name="driver_view", size=self.driver_view_panel_width),
            Layout(name="right", size=self.telemetry_panel_width)
        )
        layout["right"].split_column(
            Layout(name="top_right", size=self.lap_data_panel_height + 2),
            Layout(name="telemetry", size=self.telemetry_panel_height + 2),
            Layout(name="minimap", size=self.minimap_panel_height + 2)
        )
        layout["top_right"].split_row(
            Layout(name="lap_data", size=self.lap_data_panel_width),
            Layout(name="sector_timing", size=self.lap_data_panel_width)
        )
        lap = 1
        lap_start_dates = self.laps_data['LapStartDate']
        lap_numbers = self.laps_data['LapNumber']
        # with Live(layout, screen=False, refresh_per_second=1/self.refresh_rate):
        with Live(layout, screen=False, refresh_per_second=60):
            for i in range(len(self.telemetry)):
                start_time = datetime.now()
                clock = self.telemetry['Date'].iloc[i].to_pydatetime()
                if clock >= lap_start_dates.iloc[lap - 1]:
                    lap = int(lap_numbers.iloc[lap - 1])
                
                driver_view_frame = self.driver_view_panel.generate_frame(i)
                telemetry_frame = self.telemetry_panel.generate_frame(i)
                lap_data_frame = self.lap_data_panel.generate_frame(lap)
                sector_timing_frame = self.sector_timing_panel.generate_frame(lap, i)
                minimap_frame = self.minimap_panel.generate_frame(i)
                
                driver_view_panel = Panel(driver_view_frame, title=f"Driver View", width=self.driver_view_panel_width + 2, height=self.driver_view_panel_height + 2)
                telemetry_panel = Panel(telemetry_frame, title="Telemetry", width=self.telemetry_panel_width + 2, height=self.telemetry_panel_height + 2)
                lap_data_panel = Panel(lap_data_frame, title="Lap Data", width=self.lap_data_panel_width + 2, height=self.lap_data_panel_height + 2)
                sector_timing_panel = Panel(sector_timing_frame, title="Sector Timing", width=self.lap_data_panel_width + 2, height=self.lap_data_panel_height + 2)
                minimap_panel = Panel(minimap_frame, title="Minimap", width=self.minimap_panel_width + 2, height=self.minimap_panel_height + 2)
                layout['driver_view'].update(driver_view_panel)
                layout['telemetry'].update(telemetry_panel)
                layout['lap_data'].update(sector_timing_panel)
                layout['sector_timing'].update(lap_data_panel)
                layout['minimap'].update(minimap_panel)
                
                # time_delta = (self.telemetry['Date'].iloc[i+1] - self.telemetry['Date'].iloc[i]).total_seconds() - (datetime.now() - start_time).total_seconds()
                
                # if time_delta > 0:
                #     time.sleep(time_delta)
                
                time.sleep(self.refresh_rate)

if __name__ == "__main__":
    racetrack_database_loader = RacetrackDatabaseLoader('Silverstone')
    track_data = racetrack_database_loader.get_track_data()
    telemetry_loader = TelemetryLoader(2025, 'Silverstone', 'R')
    session = telemetry_loader.get_session()
    session.load()
    laps_data = session.laps.pick_drivers(['NOR'])
    laps_data.reset_index(drop=True)
    display = F1AsciiReplayDisplay(
        laps_data=laps_data,
        track_data = track_data
    )
    display.main()

