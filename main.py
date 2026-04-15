
import math
import shutil
import time
from datetime import datetime
from itertools import groupby

import numpy as np
import pandas as pd
from loaders import *
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

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
    s = (0, 127, 0) # FF0000 (red)
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
    def __init__(self, panel_width, panel_height, laps_data, track_data, fov, lookahead, camera_height, horizon_y):
        self.laps_data = laps_data
        self.telemetry = laps_data.telemetry
        self.track_data = track_data
        self.panel_width = panel_width
        self.panel_height = panel_height
        self.fov = fov
        self.lookahead = lookahead
        self.camera_height = camera_height
        self.horizon_y = horizon_y
        
        self.focal_length = (self.panel_width / 2) / math.tan(math.radians(self.fov) / 2)

        self.x_car = self.telemetry['X'].to_numpy()/10
        self.y_car = self.telemetry['Y'].to_numpy()/10
        self.x_track = self.track_data['x_m']
        self.y_track = self.track_data['y_m']
        self.width_left = self.track_data['w_tr_left_m']
        self.width_right = self.track_data['w_tr_right_m']
        
        self.x_track_min, self.y_track_min = self.x_track.min(), self.y_track.min()
        self.x_car_min, self.y_car_min = self.x_car.min(), self.y_car.min()
        self.x_scale = (self.x_track.max() - self.x_track.min()) / (self.x_car.max() - self.x_car.min())
        self.y_scale = (self.y_track.max() - self.y_track.min()) / (self.y_car.max() - self.y_car.min())
        
        # Stack track points
        self.track_xy = np.column_stack((self.x_track, self.y_track))

        # Compute forward differences (vectorised)
        dx = np.roll(self.x_track, -1) - self.x_track
        dy = np.roll(self.y_track, -1) - self.y_track

        # Avoid last-point glitch
        dx[-1] = self.x_track[-1] - self.x_track[-2]
        dy[-1] = self.y_track[-1] - self.y_track[-2]

        # Normalise tangents
        lengths = np.hypot(dx, dy)
        dx /= lengths
        dy /= lengths

        # Normals (perpendicular)
        self.normals = np.column_stack((-dy, dx))
    
    def _car_to_track_co_ords(self, car_x, car_y):
        return (
            self.x_track_min + (car_x - self.x_car_min) * self.x_scale,
            self.y_track_min + (car_y - self.y_car_min) * self.y_scale
        )

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
            # default to the backward difference if we're at the end of the lap data
            dx = self.x_track[j] - self.x_track[j - window]
            dy = self.y_track[j] - self.y_track[j - window]

        return math.atan2(dy, dx)

    def _build_track_points(self, i):
        cam_heading = self._calculate_car_heading(i, 1)
        cam_x, cam_y = self._car_to_track_co_ords(self.x_car[i], self.y_car[i])

        cos_h = math.cos(-cam_heading)
        sin_h = math.sin(-cam_heading)
        
        left_xy = self.track_xy + self.normals * self.width_left[:, None]
        right_xy = self.track_xy - self.normals * self.width_right[:, None]

        def project(points):
            dx = points[:, 0] - cam_x
            dy = points[:, 1] - cam_y

            # rotate points into camera view
            x_rel = dx * cos_h - dy * sin_h
            y_rel = dx * sin_h + dy * cos_h

            # remove points behind camera
            mask = y_rel > 0
            x_rel = x_rel[mask]
            y_rel = y_rel[mask]
            
            # perspective
            inv_z = 1.0 / y_rel
            screen_x = (self.panel_width / 2 + x_rel * inv_z * self.focal_length).astype(int)
            screen_y = (
                self.horizon_y * self.panel_height
                + self.camera_height * inv_z * self.panel_height
            ).astype(int)

            valid = (
                (screen_x >= 0) & (screen_x < self.panel_width) &
                (screen_y >= 0) & (screen_y < self.panel_height)
            )
            return screen_y[valid], screen_x[valid]

        cy, cx = project(self.track_xy)
        ly, lx = project(left_xy)
        ry, rx = project(right_xy)

        return (cy, cx), (ly, lx), (ry, rx)

    def _rasterise_triangle(self, buf, v1, v2, v3, char='.'):
        h = self.panel_height
        w = self.panel_width

        x1, y1 = v1[1], v1[0]
        x2, y2 = v2[1], v2[0]
        x3, y3 = v3[1], v3[0]

        # Bounding box (clamped to screen)
        min_x = max(0, int(min(x1, x2, x3)))
        max_x = min(w - 1, int(max(x1, x2, x3)))
        min_y = max(0, int(min(y1, y2, y3)))
        max_y = min(h - 1, int(max(y1, y2, y3)))

        def edge(ax, ay, bx, by, px, py):
            return (px - ax) * (by - ay) - (py - ay) * (bx - ax)

        # Rasterise
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                w1 = edge(x1, y1, x2, y2, x, y)
                w2 = edge(x2, y2, x3, y3, x, y)
                w3 = edge(x3, y3, x1, y1, x, y)

                if (w1 >= 0 and w2 >= 0 and w3 >= 0) or (w1 <= 0 and w2 <= 0 and w3 <= 0):
                    if 0 <= y < h and 0 <= x < w:
                        if buf[y][x] == ' ':
                            buf[y][x] = char

    def _project_points(self, points, cam_x, cam_y, cos_h, sin_h):
        dx = points[:, 0] - cam_x
        dy = points[:, 1] - cam_y

        x_rel = dx * cos_h - dy * sin_h
        y_rel = dx * sin_h + dy * cos_h

        mask = y_rel > 0.5
        
        inv_z = 1.0 / y_rel

        screen_x = (self.panel_width / 2 + x_rel * inv_z * self.focal_length).astype(int)
        screen_y = (
            self.horizon_y * self.panel_height
            + self.camera_height * inv_z * self.panel_height
        ).astype(int)

        valid = (
            mask &
            (screen_x >= 0) & (screen_x < self.panel_width) &
            (screen_y >= 0) & (screen_y < self.panel_height)
        )

        return screen_y, screen_x, valid
    
    def _plot(self, buf, r, c, char):
        if 0 <= r < self.panel_height and 0 <= c < (self.panel_width):
            buf[r][c] = char
    
    def generate_frame(self, i):
        buf = [[' '] * (self.panel_width) for _ in range(self.panel_height)]
        (cy, cx), (ly, lx), (ry, rx) = self._build_track_points(i)

        # cam_heading = self._calculate_car_heading(i, 1)
        # cam_x, cam_y = self._car_to_track_co_ords(self.x_car[i], self.y_car[i])

        # cos_h = math.cos(-cam_heading)
        # sin_h = math.sin(-cam_heading)

        # # World points
        # left_xy = self.track_xy + self.normals * self.width_left[:, None]
        # right_xy = self.track_xy - self.normals * self.width_right[:, None]

        # # Project ALL
        # cy, cx, c_valid = self._project_points(self.track_xy, cam_x, cam_y, cos_h, sin_h)
        # ly, lx, l_valid = self._project_points(left_xy, cam_x, cam_y, cos_h, sin_h)
        # ry, rx, r_valid = self._project_points(right_xy, cam_x, cam_y, cos_h, sin_h)

        # buf = [[' '] * (self.panel_width) for _ in range(self.panel_height)]

        # n = len(self.track_xy)

        # for j in range(n - 1):
        #     # Skip if any of the 4 points are invalid
        #     if not (l_valid[j] or r_valid[j] or l_valid[j+1] or r_valid[j+1]):
        #         continue
            
        #     L1 = (ly[j], lx[j])
        #     R1 = (ry[j], rx[j])
        #     L2 = (ly[j+1], lx[j+1])
        #     R2 = (ry[j+1], rx[j+1])
            
        #     # Rasterise 2 triangles
        #     self._rasterise_triangle(buf, L1, R1, L2)
        #     self._rasterise_triangle(buf, R1, R2, L2)
        
        
        # . for centre line, | for track edges
        for r, c in zip(cy, cx):
            self._plot(buf, r, c, '"')
        
        for r, c in zip(ly, lx):
            self._plot(buf, r, c, '|')

        for r, c in zip(ry, rx):
            self._plot(buf, r, c, '|')
        
        return '\n'.join(''.join(row) for row in buf)


class TelemetryAsciiPanel:
    def __init__(self, panel_width, panel_height, laps_data):
        self.panel_width = panel_width
        self.panel_height = panel_height
        self.laps_data = laps_data
        self.telemetry = laps_data.telemetry
        
        self.speed_arr = self.telemetry['Speed'].to_numpy()
        self.rpm_arr = self.telemetry['RPM'].to_numpy()
        self.throttle_arr = self.telemetry['Throttle'].to_numpy()
        self.brake_arr = self.telemetry['Brake'].to_numpy()
        self.ngear_arr = self.telemetry['nGear'].to_numpy()
        self.drs_arr = self.telemetry['DRS'].to_numpy()
        self.date_arr = self.telemetry['Date'].to_numpy()
        self.session_time_arr = self.telemetry['SessionTime'].to_numpy()
        
        self.bar_colours = _generate_throttle_gradient(20)
        self.gears = ['N', '1', '2', '3', '4', '5', '6', '7', '8']
        
        # initalise speedometer
        self.speedometer_width = 32
        self.speedometer_height = 3
        self.speedometer_perimeter = self.speedometer_width + self.speedometer_height * 2 - 2
        self.grid = [[" " for _ in range(self.speedometer_width)] for _ in range(self.speedometer_height)]
        self.path = []
        # 1. LEFT column (bottom → top)
        for y in range(self.speedometer_height - 1, 0, -1):
            self.path.append((y, 0))
        # 2. TOP row (left → right)
        for x in range(0, self.speedometer_width):
            self.path.append((0, x))
        # 3. RIGHT column (top → bottom)
        for y in range(1, self.speedometer_height):
            self.path.append((y, self.speedometer_width - 1))

        # Draw border
        for i, (y, x) in enumerate(self.path):
            # Choose char based on position
            if y == 0 and x == 0:
                self.grid[y][x] = '╭'
            elif y == 0 and x == self.speedometer_width - 1:
                self.grid[y][x] = '╮'
            elif y == self.speedometer_height-1 and x == 0:
                self.grid[y][x] = '0'
            elif y == self.speedometer_height-1 and x == self.speedometer_width - 1:
                self.grid[y][x] = 'x'
            elif y == 0:
                self.grid[y][x] = '─'
            else:
                self.grid[y][x] = '│'
                

    def _render_throttle_brake_bar(self, throttle_percent, brake, bar_length=20):
        return ['[#FF0000]<[/]']*bar_length if brake else [f'[bold {self.bar_colours[idx]}]>[/]' if idx <= throttle_percent // (100/len(self.bar_colours)) else '>' for idx in range(len(self.bar_colours))]
    
    def _render_rpm_bar(self, rpm, bar_length=20, max_rpm=15000, rpm_red_point=10000):
        # Three sections of the bar - blue and red for the RPM and blank. 
        # Red is for above 10000RPM, chosen to estimate the "upshift" RPM value. Change this ad lib.
        rpm_bar = f'[blue]{"●"*int(rpm/max_rpm * bar_length)}{"○"*(bar_length-int((rpm)/max_rpm * bar_length))}[/]'
        red_index = 6 + int(rpm_red_point/max_rpm * bar_length) - 1
        return rpm_bar[:red_index] + '[/][red]' + rpm_bar[red_index:]
    
    def _render_gear(self, n_gear):
        n_gear = str(n_gear) if n_gear != 0 else 'N'
        return ' '.join(
            [f'[grey]{num}[/]' if num != n_gear else f'[blue bold underline]{num}[/]'
             for num in self.gears
             ])
    
    def _accel_curve(self, p, curve_index=2):
        # 2 -> quadratic
        return 1 - (1 - p) ** curve_index
    
    def _render_speedometer(self, speed, throttle, brake):
        # shallow copy per row
        grid_buffer = [row[:] for row in self.grid]
        filled = int(self._accel_curve(min(speed / 350, 1)) * self.speedometer_perimeter)
        for i, (y, x) in enumerate(self.path):
            if i < filled:
                grid_buffer[y][x] = f'[green]{grid_buffer[y][x]}[/]'

        # Speed text in center
        speed_str = f"{int(speed)} km/h"
        cx = (self.speedometer_width - len(speed_str)) // 2
        throttle_just = str(throttle).rjust(2, ' ')
        throttle_brake_bar = self._render_throttle_brake_bar(throttle, brake)
        throttle_brake_str = (
            [' ']*5 + throttle_brake_bar + [' ', '[red]B', 'R', 'K[/]'] 
        ) if brake else (
            [' ', f'[green]{throttle_just[0]}', throttle_just[1], '%[/]', ' '] + throttle_brake_bar)
        
        for i, char in enumerate(speed_str):
            grid_buffer[1][cx + i] = char

        for i, char in enumerate(throttle_brake_str):
            grid_buffer[2][1 + i] = char

        return '\n'.join([''.join(row) for row in grid_buffer])
    
    def generate_frame(self, i):
        speed = self.speed_arr[i]
        rpm = int(self.rpm_arr[i])
        throttle_percent = int(self.throttle_arr[i])
        brake = self.brake_arr[i]
        n_gear = self.ngear_arr[i]
        drs = DRS_KEY[self.drs_arr[i]]
        date = self.date_arr[i]
        session_time = self.session_time_arr[i]
        
        rpm_bar = self._render_rpm_bar(rpm)
        speedometer = self._render_speedometer(speed, throttle_percent, brake)
        gear_display = self._render_gear(n_gear)
        
        return f"""{speedometer} DRS: {drs}
GEAR: {gear_display}
RPM |{rpm_bar}| {rpm}
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
        gx = int((x - self.x_car_min) / (self.x_car_max - self.x_car_min) * (self.panel_width - 10)) + 5
        gy = self.panel_height - 1 - int((y - self.y_car_min) / (self.y_car_max - self.y_car_min) * (self.panel_height - 6)) - 3
        return gx, gy

    def _track_to_screen(self, x, y):
        gx = int((x - self.x_track_min) / (self.x_track_max - self.x_track_min) * (self.panel_width - 10)) + 5
        gy = self.panel_height - 1 - int((y - self.y_track_min) / (self.y_track_max - self.y_track_min) * (self.panel_height - 6)) - 3
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
            self.track_map_cache = [row[:] for row in buf]
        else:
            buf = [row[:] for row in self.track_map_cache]

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
    def __init__(self, telemetry_loader, racetrack_database_loader, terminal_width=None, terminal_height=None, fov=60.0, lookahead=50.0, camera_height=10, horizon_y=0.2, refresh_rate=1/30):
        self.refresh_rate = refresh_rate
        
        track_data = racetrack_database_loader.get_track_data()
        session = telemetry_loader.get_session()
        session.load()
        self.laps_data = session.laps.pick_drivers(['ALB'])
        self.laps_data.reset_index(drop=True)
        corners = session.get_circuit_info().corners
        race_control_messages = session.race_control_messages
        gmt_offset = session.session_info['GmtOffset']
        
        # These dimensions have been mostly judged arbitrarily in terms of ratios - the 8 line high windows are to match the amount of space the given data takes up, 
        # and the subtractions are for border thicknesses. The exact workings on this I'm lost on, blame Rich.
        
        self.terminal_width = terminal_width or (shutil.get_terminal_size()[0])
        self.terminal_height = terminal_height or (shutil.get_terminal_size()[1])
        
        self.race_control_messages_ascii_panel_width = int((self.terminal_width) * 0.65) - 4
        self.race_control_messages_ascii_panel_height = 4
        
        self.driver_view_ascii_panel_width = int((self.terminal_width) * 0.65) - 4
        self.driver_view_ascii_panel_height = self.terminal_height - self.race_control_messages_ascii_panel_height - 9 - 4
        
        self.lap_data_ascii_panel_width = self.driver_view_ascii_panel_width // 2 - 2
        self.lap_data_ascii_panel_height = 7
        
        self.telemetry_ascii_panel_width = self.driver_view_ascii_panel_width - self.lap_data_ascii_panel_width - 4
        self.telemetry_ascii_panel_height = 7
        
        self.sector_timing_ascii_panel_width = self.terminal_width - self.race_control_messages_ascii_panel_width - 8
        self.sector_timing_ascii_panel_height = 8
        
        self.minimap_ascii_panel_width = self.terminal_width - self.race_control_messages_ascii_panel_width - 8
        self.minimap_ascii_panel_height = self.terminal_height - self.sector_timing_ascii_panel_height - 4
        
        self.driver_view_ascii_panel = DriverViewAsciiPanel(
            panel_width=self.driver_view_ascii_panel_width,
            panel_height=self.driver_view_ascii_panel_height,
            laps_data=self.laps_data,
            track_data=track_data,
            fov=fov,
            lookahead=lookahead,
            camera_height=camera_height,
            horizon_y=horizon_y
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
            Layout(name="left", size=self.race_control_messages_ascii_panel_width + 4),
            Layout(name="right", size=self.sector_timing_ascii_panel_width + 4)
        )
        layout["left"].split_column(
            Layout(name='race_control_messages', size=self.race_control_messages_ascii_panel_height + 2),
            Layout(name='driver_view', size=self.driver_view_ascii_panel_height + 2),
            Layout(name='bottom_left', size=self.lap_data_ascii_panel_height + 2)
        )
        layout['bottom_left'].split_row(
            Layout(name="lap_data", size=self.lap_data_ascii_panel_width + 4),
            Layout(name="telemetry", size=self.telemetry_ascii_panel_width + 4),
        )
        layout["right"].split_column(
            Layout(name="sector_timing", size=self.sector_timing_ascii_panel_height + 2),
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
                
                driver_view_panel = Panel(driver_view_frame, title="Driver View", width=self.driver_view_ascii_panel_width + 4, height=self.driver_view_ascii_panel_height + 2)
                lap_data_panel = Panel(lap_data_frame, title="Lap Data", width=self.lap_data_ascii_panel_width + 4, height=self.lap_data_ascii_panel_height + 2)
                sector_timing_panel = Panel(sector_timing_frame, title="Sector Timing", width=self.sector_timing_ascii_panel_width + 4, height=self.sector_timing_ascii_panel_height + 2)
                telemetry_panel = Panel(telemetry_frame, title="Telemetry", width=self.telemetry_ascii_panel_width + 4, height=self.telemetry_ascii_panel_height + 2)
                minimap_panel = Panel(minimap_frame, title="Minimap", width=self.minimap_ascii_panel_width + 4, height=self.minimap_ascii_panel_height + 2)
                race_control_messages_panel = Panel(race_control_messages_frame, title="Race Control", width=self.race_control_messages_ascii_panel_width + 4, height=self.race_control_messages_ascii_panel_height + 2)
                
                layout['driver_view'].update(driver_view_panel)
                layout['sector_timing'].update(sector_timing_panel)
                layout['lap_data'].update(lap_data_panel)
                layout['telemetry'].update(telemetry_panel)
                layout['minimap'].update(minimap_panel)
                layout['race_control_messages'].update(race_control_messages_panel)
                
                # time_delta = (self.telemetry['Date'].iloc[i+1] - self.telemetry['Date'].iloc[i]).total_seconds() - (datetime.now() - start_time).total_seconds()
                
                # if time_delta > 0:
                #     time.sleep(time_delta)
                
                elapsed = (datetime.now() - start_time).total_seconds()
                sleep_time = max(0, self.refresh_rate - elapsed)
                time.sleep(sleep_time)

if __name__ == "__main__":
    telemetry_loader = TelemetryLoader(2025, 'Silverstone', 'R')
    racetrack_database_loader = RacetrackDatabaseLoader('Silverstone')
    display = F1AsciiReplayDisplay(
        telemetry_loader=telemetry_loader,
        racetrack_database_loader=racetrack_database_loader,
        refresh_rate=1/30
    )
    display.main()

