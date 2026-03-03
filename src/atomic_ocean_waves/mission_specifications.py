"""
ATOMIC mission times and locations.
"""

import pandas as pd

# Mission time periods

leg_1_times = (
    pd.Timestamp('2020-01-08T00:00', tz='utc'),
    pd.Timestamp('2020-01-25T00:00', tz='utc'),
)
leg_2_times = (
    pd.Timestamp('2020-01-28T00:00', tz='utc'),
    pd.Timestamp('2020-02-13T00:00', tz='utc'),
)

# ((lon_min, lon_max), lat_min, lat_max)
leg_1_coordinates = (
    (-62, -50), (12.5, 16.5)
)
leg_2_coordinates = (
    (-62, -50), (12.5, 16.5)
)

# leg_1_time_start = pd.Timestamp('2020-01-08T00:00', tz='utc')
# leg_1_time_end = pd.Timestamp('2020-01-25T00:00', tz='utc')
# leg_2_time_start = pd.Timestamp('2020-01-28T00:00', tz='utc')
# leg_2_time_end = pd.Timestamp('2020-02-13T00:00', tz='utc')

# Station locations