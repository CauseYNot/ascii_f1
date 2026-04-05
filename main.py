
import math
import shutil
import time
from datetime import datetime
from itertools import groupby

import pandas as pd
from loaders import *
# from rich.box import box
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from copy import deepcopy

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
    def __init__(self, panel_width, panel_height, laps_data, track_data, fov, lookahead, camera_height, horizon_y, vertical_scale):
        self.laps_data = laps_data
        self.telemetry = laps_data.telemetry
        self.track_data = track_data
        self.panel_width = panel_width
        self.panel_height = panel_height
        self.fov = fov
        self.lookahead = lookahead
        self.camera_height = camera_height
        self.horizon_y = horizon_y
        self.vertical_scale = vertical_scale

        self.x_car = self.telemetry['X'].to_numpy()/10
        self.y_car = self.telemetry['Y'].to_numpy()/10
        self.x_track = self.track_data['x_m']
        self.y_track = self.track_data['y_m']
        self.width_left = self.track_data['w_tr_left_m']
        self.width_right = self.track_data['w_tr_right_m']
        
        self.x_car_min, self.x_car_max = self.x_car.min(), self.x_car.max()
        self.y_car_min, self.y_car_max = self.y_car.min(), self.y_car.max()
        self.x_track_min, self.x_track_max = self.x_track.min(), self.x_track.max()
        self.y_track_min, self.y_track_max = self.y_track.min(), self.y_track.max()
        
    #     # convert FastF1 telemetry to metres
    #     car_pts = np.column_stack([
    #         self.x_car * 0.1,
    #         self.y_car * 0.1
    #     ])

    #     track_pts = np.column_stack([
    #         self.x_track,
    #         self.y_track
    #     ])

    #     # resample to equal length
    #     N = 500
    #     car_sample = car_pts[np.linspace(0, len(car_pts)-1, N).astype(int)]
    #     track_sample = track_pts[np.linspace(0, len(track_pts)-1, N).astype(int)]

    #     self.scale, self.R, self.t = self._compute_similarity_transform(
    #         car_sample, track_sample
    #     )

    # def _compute_similarity_transform(self, src, dst):
    #     src_mean = src.mean(axis=0)
    #     dst_mean = dst.mean(axis=0)

    #     src_c = src - src_mean
    #     dst_c = dst - dst_mean

    #     cov = dst_c.T @ src_c / len(src)
    #     U, S, Vt = np.linalg.svd(cov)

    #     R = U @ Vt
    #     if np.linalg.det(R) < 0:
    #         Vt[-1, :] *= -1
    #         R = U @ Vt

    #     scale = np.trace(np.diag(S)) / np.sum(src_c**2)
    #     t = dst_mean - scale * (R @ src_mean)

    #     return scale, R, t

    # def _car_to_track_co_ords(self, car_x, car_y):
    #     p = np.array([car_x * 0.1, car_y * 0.1])  # dm → m
    #     return tuple(self.scale * (self.R @ p) + self.t)
    
    def _car_to_track_co_ords(self, car_x, car_y):
        x_car_fraction = (car_x - self.x_car_min) / (self.x_car_max - self.x_car_min)
        x_track_converted = self.x_track_min + (x_car_fraction) * (self.x_track_max - self.x_track_min)
        y_car_fraction = (car_y - self.y_car_min) / (self.y_car_max - self.y_car_min)
        y_track_converted = self.y_track_min + (y_car_fraction) * (self.y_track_max - self.y_track_min)

        return x_track_converted, y_track_converted

    def _calculate_car_heading(self, i, window=1):
        # takes the angle between points [window] ahead and behind of the current telemetry position. defaults to beginning/end of telemetry points.
        i0 = max(0, i - window)
        i1 = min(len(self.x_car) - 1, i + window)

        dx = self.x_car[i1] - self.x_car[i0]
        dy = self.y_car[i1] - self.y_car[i0]

        return math.atan2(dy, dx) - math.pi/2 # only the god of coding understands why a π/2 rotation is necessary. I don't.
    
    def _calculate_track_heading(self, j, window=1):
        # estimate track tangent using forward difference
        if j < len(self.x_track) - 1:
            dx = self.x_track[j + window] - self.x_track[j]
            dy = self.y_track[j + window] - self.y_track[j]
        else:
            # default to the backward difference if we're at the end of the lap
            dx = self.x_track[j] - self.x_track[j - window]
            dy = self.y_track[j] - self.y_track[j - window]

        return math.atan2(dy, dx)

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
        
        # if not (0 < y_rel < self.lookahead):
        #     return None

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

    def _lod_step(self, y_rel):
        if y_rel < 0.2* self.lookahead:
            return 1      # very near
        elif y_rel < 0.4*self.lookahead:
            return 2
        elif y_rel < 0.7*self.lookahead:
            return 4
        else:
            return 16      # far away
    
    def _build_track_points(self, i):
        left_points = []
        centre_points = []
        right_points = []

        cam_heading = self._calculate_car_heading(i, 1)
        cam_x, cam_y = self._car_to_track_co_ords(self.x_car[i], self.y_car[i])

        j = 0
        while j < len(self.x_track):
            xt = self.x_track[j]
            yt = self.y_track[j]
            wl = self.width_left[j]
            wr = self.width_right[j]
            
            track_heading = self._calculate_track_heading(j, 1)
            normal_x = -math.sin(track_heading)
            normal_y =  math.cos(track_heading)

            x_left, y_left = xt + (normal_x * wl), yt + (normal_y * wl)
            x_right, y_right = xt - (normal_x * wr), yt - (normal_y * wr)

            left_point = self._project_point(x_left, y_left, cam_x, cam_y, cam_heading)
            centre_point = self._project_point(xt, yt, cam_x, cam_y, cam_heading)
            right_point = self._project_point(x_right, y_right, cam_x, cam_y, cam_heading)

            left_points.append(left_point)
            centre_points.append(centre_point)
            right_points.append(right_point)
            
            dx = xt - cam_x
            dy = yt - cam_y
            j += self._lod_step(y_rel=(dx * math.sin(cam_heading) + dy * math.cos(cam_heading)))

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
    def __init__(self, panel_width, panel_height, laps_data):
        self.panel_width = panel_width
        self.panel_height = panel_height
        self.laps_data = laps_data
        self.telemetry = laps_data.telemetry

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


class LapDataAsciiPanel:
    def __init__(self, panel_width, panel_height, laps_data):
        self.panel_width = panel_width
        self.panel_height = panel_height
        self.laps_data = laps_data
    
    def _tyre_strategy_diagram(self, lap):
        compounds = list(self.laps_data['Compound'])[:lap]
        stints = [[compound, len(list(group))] for compound, group in groupby(compounds)]
        shrink_factor = (self.panel_width - 30) / len(self.laps_data)
        return '|'.join(f"[{TYRE_KEY[stint[0]]}]{"█"*max(1, int(stint[1]*shrink_factor))} {stint[1]} [/]" for stint in stints)
    
    def generate_frame(self, lap):
        lap_data = self.laps_data.iloc[lap]
        return f"""[bold underline]Driver:[/] [{CONSTRUCTOR_COLOUR_KEY[lap_data['Team']]}]{lap_data['DriverNumber']} - {lap_data['Driver']} ({lap_data['Team']})[/]
[bold underline]Lap: {lap}[/]
Tyre Compound: [{TYRE_KEY[lap_data['Compound']]}]{lap_data['Compound']}[/]
Tyre Age: {int(lap_data['TyreLife'])}
Stint: {int(lap_data['Stint'])}
Position: P{int(lap_data['Position'])}
Tyre Strategy: |{self._tyre_strategy_diagram(lap=lap)}"""


class SectorTimingAsciiPanel:
    def __init__(self, panel_width, panel_height, laps_data):
        self.panel_width = panel_width
        self.panel_height = panel_height
        self.laps_data = laps_data
        self.telemetry = laps_data.telemetry
    
    def _format_time_with_colour(self, t, best):
        if pd.isna(t):
            return "--.---"
        if isinstance(t, str):
            return t

        total_seconds = t.total_seconds()

        minutes = int(total_seconds // 60)
        seconds = total_seconds % 60

        if minutes > 0:
            t_str = f"{minutes}:{seconds:06.3f}"
        else:
            t_str = f"{seconds:6.3f}"
        
        if t < best or pd.isna(best):
            return f'[bold green]{t_str}[/]'
        else:
            return f'[bold yellow]{t_str}[/]'
        
    
    def _format_time(self, t):
        if pd.isna(t):
            return "--.---"
        if isinstance(t, str):
            return t

        total_seconds = t.total_seconds()

        minutes = int(total_seconds // 60)
        seconds = total_seconds % 60

        if minutes > 0:
            t_str = f"{minutes}:{seconds:06.3f}"
        else:
            t_str = f"{seconds:6.3f}"
        
        return t_str
        

    def generate_frame(self, lap, i):
        current_telemetry_data = self.telemetry.iloc[i]
        lap_data = self.laps_data.iloc[lap - 1]
        lap_start_time = lap_data['LapStartTime']
        best_sector_1_time = min(self.laps_data['Sector1Time'].iloc[:lap-1].dropna(), default=pd.NaT)
        best_sector_2_time = min(self.laps_data['Sector2Time'].iloc[:lap-1].dropna(), default=pd.NaT)
        best_sector_3_time = min(self.laps_data['Sector3Time'].iloc[:lap-1].dropna(), default=pd.NaT)
        best_lap_time = min(self.laps_data['LapTime'].iloc[:lap-1].dropna(), default=pd.NaT)
        current_lap_time = current_telemetry_data['SessionTime'] - lap_start_time
        sector_1_time = lap_data['Sector1Time']
        sector_2_time = lap_data['Sector2Time'] if not pd.isna(sector_1_time) else pd.NaT
        sector_3_time = lap_data['Sector3Time'] if not pd.isna(sector_2_time) else pd.NaT
        
        if current_lap_time < sector_1_time:
            sector_1_time, sector_2_time, sector_3_time = current_lap_time, pd.NaT, pd.NaT
        elif current_lap_time < (sector_1_time + sector_2_time):
            sector_2_time, sector_3_time = current_lap_time - sector_1_time, pd.NaT
        elif current_lap_time < (sector_1_time + sector_2_time + sector_3_time):
            sector_3_time = current_lap_time - sector_2_time - sector_1_time
        
        table = Table(expand=True)
        table.add_column(header='Sector')
        table.add_column(header='Time')
        table.add_column(header='Best')
        table.add_row("S1", self._format_time_with_colour(sector_1_time, best_sector_1_time), self._format_time(best_sector_1_time))
        table.add_row("S2", self._format_time_with_colour(sector_2_time, best_sector_2_time), self._format_time(best_sector_2_time))
        table.add_row("S3", self._format_time_with_colour(sector_3_time, best_sector_3_time), self._format_time(best_sector_3_time))
        table.add_row("Lap Time", self._format_time_with_colour(current_lap_time, best_lap_time), self._format_time(best_lap_time))
        
        return table


class MinimapAsciiPanel:
    def __init__(self, panel_width, panel_height, laps_data, track_data, corners=None):
        self.panel_width = panel_width
        self.panel_height = panel_height
        self.laps_data = laps_data
        self.telemetry = laps_data.telemetry
        self.track_data = track_data
        self.corners = corners
        
        self.track_map_cache = None

        self.x_car = self.telemetry['X'].to_numpy()
        self.y_car = self.telemetry['Y'].to_numpy()
        self.x_track = self.track_data['x_m']
        self.y_track = self.track_data['y_m']
        
        # Normalisation bounds
        self.x_car_min, self.x_car_max = self.x_car.min(), self.x_car.max()
        self.y_car_min, self.y_car_max = self.y_car.min(), self.y_car.max()
        self.x_track_min, self.x_track_max = self.x_track.min(), self.x_track.max()
        self.y_track_min, self.y_track_max = self.y_track.min(), self.y_track.max()

    # ---------- Coordinate transforms ----------
    def _tel_to_screen(self, x, y):
        gx = int((x - self.x_car_min) / (self.x_car_max - self.x_car_min) * (self.panel_width - 20)) + 10
        gy = self.panel_height - 1 - int((y - self.y_car_min) / (self.y_car_max - self.y_car_min) * (self.panel_height - 10)) - 5
        return gx, gy

    def _track_to_screen(self, x, y):
        gx = int((x - self.x_track_min) / (self.x_track_max - self.x_track_min) * (self.panel_width - 20)) + 10
        gy = self.panel_height - 1 - int((y - self.y_track_min) / (self.y_track_max - self.y_track_min) * (self.panel_height - 10)) - 5
        return gx, gy
    
    def _rotate(self, xy, *, angle):
        rot_mat = np.array([[np.cos(angle), np.sin(angle)],
                            [-np.sin(angle), np.cos(angle)]])
        
        return np.matmul(xy, rot_mat)
    
    def _draw_corner_numbers(self, buf, corners):
        for _, corner in corners.iterrows():
            cx = corner['X']
            cy = corner['Y']
            angle = corner['Angle']
            number = int(corner['Number'])

            theta = math.radians(angle)
            OFFSET = 500.0  # decimetres
            vector = [OFFSET, 0]
            nx, ny = self._rotate(vector, angle=theta)

            lx = cx + nx
            ly = cy + ny

            sx, sy = self._tel_to_screen(lx, ly)
            label = str(number)
            start_x = sx - len(label) // 2

            for i, ch in enumerate(label):
                x = start_x + i
                if 0 <= x < self.panel_width and 0 <= sy < self.panel_height:
                    buf[sy][x] = f"[bold yellow]{ch}[/bold yellow]"

    # ---------- Frame generation ----------
    def generate_frame(self, i):
        if self.track_map_cache is None:
            buf = [[" "] * self.panel_width for _ in range(self.panel_height)]

            # Draw track centreline
            for k in range(len(self.x_track)):
                cx, cy = self._track_to_screen(self.x_track[k], self.y_track[k])
                
                if 0 <= cx < self.panel_width and 0 <= cy < self.panel_height:
                    buf[cy][cx] = "#"

            self._draw_corner_numbers(buf, self.corners)
            self.track_map_cache = deepcopy(buf)
        else:
            buf = deepcopy(self.track_map_cache)

        # Draw car position
        gx, gy = self._tel_to_screen(self.x_car[i], self.y_car[i])
        if 0 <= gx < self.panel_width and 0 <= gy < self.panel_height:
            colour = CONSTRUCTOR_COLOUR_KEY[self.laps_data['Team'].iloc[0]]
            buf[gy][gx] = f"[{colour}]●[/]"

        return "\n".join("".join(row) for row in buf)

class RaceControlMessagesAsciiPanel:
    def __init__(self, panel_width, panel_height, laps_data, race_control_messages, gmt_offset, message_time_length=300):
        self.panel_width = panel_width
        self.panel_height = panel_height
        self.laps_data = laps_data
        self.telemetry_dates = laps_data.telemetry['Date']
        self.race_control_messages = race_control_messages
        self.gmt_offset = gmt_offset
        
        self.message_time_length = message_time_length
        
        self.message_idx = 0
        self.last_message_frame = -1 - message_time_length
        self.last_message_text = ""
    
    def generate_frame(self, i):
        telemetry_date = self.telemetry_dates.iloc[i]
        # api's column names are inconsistent: the Date column in telemetry corresponds to the Time column in the race_control_messages df.
        if telemetry_date > self.race_control_messages['Time'].iloc[self.message_idx]:
            message_data = self.race_control_messages.iloc[self.message_idx]
            self.message_idx += 1
            self.last_message_text = f"Lap {message_data['Lap']}, Time: {message_data['Time'] + self.gmt_offset}\n{message_data['Message']}"
            return self.last_message_text
        elif i < self.last_message_frame + self.message_time_length:
            return self.last_message_text
        else:
            return ""

class F1AsciiReplayDisplay:
    def __init__(self, telemetry_loader, racetrack_database_loader, terminal_width=None, terminal_height=None, fov=40.0, lookahead=50.0, camera_height=0.5, horizon_y=0.1, vertical_scale=1.2, refresh_rate=1/30):
        self.refresh_rate = refresh_rate
        
        track_data = racetrack_database_loader.get_track_data()
        session = telemetry_loader.get_session()
        session.load()
        self.laps_data = session.laps.pick_drivers(['NOR'])
        self.laps_data.reset_index(drop=True)
        corners = session.get_circuit_info().corners
        race_control_messages = session.race_control_messages
        gmt_offset = session.session_info['GmtOffset']
        
        self.terminal_width = terminal_width or (shutil.get_terminal_size()[0])
        self.terminal_height = terminal_height or (shutil.get_terminal_size()[1])
        
        self.race_control_messages_ascii_panel_width = int((self.terminal_width - 2) * 0.65)
        self.race_control_messages_ascii_panel_height = 4
        
        self.driver_view_ascii_panel_width = int((self.terminal_width - 2) * 0.65)
        self.driver_view_ascii_panel_height = self.terminal_height - self.race_control_messages_ascii_panel_height - 4
        
        self.telemetry_ascii_panel_width = self.terminal_width - self.driver_view_ascii_panel_width
        self.telemetry_ascii_panel_height = 8
        
        self.sector_timing_ascii_panel_width = self.terminal_width - self.driver_view_ascii_panel_width
        self.sector_timing_ascii_panel_height = 8
        
        self.lap_data_ascii_panel_width = self.terminal_width - self.driver_view_ascii_panel_width
        self.lap_data_ascii_panel_height = 8
        
        self.minimap_ascii_panel_width = self.terminal_width - self.driver_view_ascii_panel_width
        self.minimap_ascii_panel_height = self.terminal_height - self.sector_timing_ascii_panel_height - self.lap_data_ascii_panel_height - self.telemetry_ascii_panel_height - 8
        
        self.driver_view_ascii_panel = DriverViewAsciiPanel(
            panel_width=self.driver_view_ascii_panel_width,
            panel_height=self.driver_view_ascii_panel_height,
            laps_data=self.laps_data,
            track_data=track_data,
            fov=fov,
            lookahead=lookahead,
            camera_height=camera_height,
            horizon_y=horizon_y,
            vertical_scale=vertical_scale
        )
        
        self.telemetry_ascii_panel = TelemetryAsciiPanel(
            panel_width=self.telemetry_ascii_panel_width,
            panel_height=self.telemetry_ascii_panel_height,
            laps_data=self.laps_data
        )
        
        self.lap_data_ascii_panel = LapDataAsciiPanel(
            panel_width=self.lap_data_ascii_panel_width,
            panel_height=self.lap_data_ascii_panel_height,
            laps_data=self.laps_data,
        )
        
        self.sector_timing_ascii_panel = SectorTimingAsciiPanel(
            panel_width=self.sector_timing_ascii_panel_width,
            panel_height=self.sector_timing_ascii_panel_height,
            laps_data=self.laps_data
        )
        
        self.minimap_ascii_panel = MinimapAsciiPanel(
            panel_width=self.minimap_ascii_panel_width,
            panel_height=self.minimap_ascii_panel_height,
            laps_data=self.laps_data,
            track_data=track_data,
            corners=corners
        )
        
        self.race_control_messages_ascii_panel = RaceControlMessagesAsciiPanel(
            panel_width=self.race_control_messages_ascii_panel_width,
            panel_height=self.race_control_messages_ascii_panel_height,
            laps_data=self.laps_data,
            race_control_messages=race_control_messages,
            gmt_offset=gmt_offset
        )
    
    def main(self):
        layout = Layout()
        layout.split_row(
            Layout(name="left", size=self.race_control_messages_ascii_panel_width),
            Layout(name="right", size=self.telemetry_ascii_panel_width)
        )
        layout["left"].split_column(
            Layout(name='race_control_messages', size=self.race_control_messages_ascii_panel_height + 2),
            Layout(name='driver_view', size=self.driver_view_ascii_panel_height + 2)
        )
        layout["right"].split_column(
            Layout(name="sector_timing", size=self.sector_timing_ascii_panel_height + 2),
            Layout(name="lap_data", size=self.lap_data_ascii_panel_height + 2),
            Layout(name="telemetry", size=self.telemetry_ascii_panel_height + 2),
            Layout(name="minimap", size=self.minimap_ascii_panel_height + 2)
        )
        lap = 1
        lap_start_dates = self.laps_data['LapStartDate']
        lap_numbers = self.laps_data['LapNumber']
        telemetry = self.laps_data.telemetry
        with Live(layout, screen=False, refresh_per_second=1/self.refresh_rate):
            for i in range(len(telemetry)):
                start_time = datetime.now()
                clock = telemetry['Date'].iloc[i].to_pydatetime()
                # checks if time has passed the start time for the next lap, then increments if necessary. relies on lap data to increment.
                if clock >= lap_start_dates.iloc[lap]:
                    lap = int(lap_numbers.iloc[lap])
                
                driver_view_frame = self.driver_view_ascii_panel.generate_frame(i)
                lap_data_frame = self.lap_data_ascii_panel.generate_frame(lap)
                sector_timing_frame = self.sector_timing_ascii_panel.generate_frame(lap, i)
                telemetry_frame = self.telemetry_ascii_panel.generate_frame(i)
                minimap_frame = self.minimap_ascii_panel.generate_frame(i)
                race_control_messages_frame = self.race_control_messages_ascii_panel.generate_frame(i)
                
                driver_view_panel = Panel(driver_view_frame, title="Driver View", width=self.driver_view_ascii_panel_width + 2, height=self.driver_view_ascii_panel_height + 2)
                lap_data_panel = Panel(lap_data_frame, title="Lap Data", width=self.lap_data_ascii_panel_width + 2, height=self.lap_data_ascii_panel_height + 2)
                sector_timing_panel = Panel(sector_timing_frame, title="Sector Timing", width=self.sector_timing_ascii_panel_width + 2, height=self.sector_timing_ascii_panel_height + 2)
                telemetry_panel = Panel(telemetry_frame, title="Telemetry", width=self.telemetry_ascii_panel_width + 2, height=self.telemetry_ascii_panel_height + 2)
                minimap_panel = Panel(minimap_frame, title="Minimap", width=self.minimap_ascii_panel_width + 2, height=self.minimap_ascii_panel_height + 2)
                race_control_messages_panel = Panel(race_control_messages_frame, title="Race Control", width=self.race_control_messages_ascii_panel_width + 2, height=self.race_control_messages_ascii_panel_height + 2)
                
                layout['driver_view'].update(driver_view_panel)
                layout['sector_timing'].update(lap_data_panel)
                layout['lap_data'].update(sector_timing_panel)
                layout['telemetry'].update(telemetry_panel)
                layout['minimap'].update(minimap_panel)
                layout['race_control_messages'].update(race_control_messages_panel)
                
                # time_delta = (self.telemetry['Date'].iloc[i+1] - self.telemetry['Date'].iloc[i]).total_seconds() - (datetime.now() - start_time).total_seconds()
                
                # if time_delta > 0:
                #     time.sleep(time_delta)
                
                time.sleep(self.refresh_rate)

if __name__ == "__main__":
    telemetry_loader = TelemetryLoader(2025, 'Silverstone', 'R')
    racetrack_database_loader = RacetrackDatabaseLoader('Silverstone')
    display = F1AsciiReplayDisplay(
        telemetry_loader=telemetry_loader,
        racetrack_database_loader=racetrack_database_loader,
        refresh_rate=1/30
    )
    display.main()

