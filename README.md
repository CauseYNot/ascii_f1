# Ascii F1
`racetrack-database` is data from the following repository: https://github.com/TUMFTM/racetrack-database
```
./
├── __pycache__/
│   └── loaders.cpython-313.pyc
├── racetrack-database/
│   ├── racelines/ 
│   ├── tracks/
│   ├── LICENSE
│   └── README.md
├── README.md
├── loaders.py
├── main.py
└── requirements.txt
```

### Usage:
Install all libraries in `requirements.txt`, then run `main.py`. Edit the race data at the bottom of the file. The display contains panels with:
- A projected view of the track from the driver's perspective
- A minimap with the position of the driver on the track
- Live telemetry data of the driver including throttle, brake, DRS position, etc.
- Lap data, including current tyre compound, tyre age, tyre strategy, stint, position, etc.
- Sector timings, with colours.