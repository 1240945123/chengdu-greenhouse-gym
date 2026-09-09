from typing import Tuple, SupportsFloat
from os.path import join

import calendar
from copy import deepcopy
from datetime import datetime, timedelta

import numpy as np
from pandas._typing import ArrayLike
import pandas as pd
from scipy.interpolate import PchipInterpolator
from glassgym.core.types import DisturbanceSchemaV2

def init_state(d0, rhMax=90, time_in_days=0):
    # Initialize the state as a NumPy array of zeros with the appropriate size
    state = np.zeros(28)  # Assuming 28 elements based on the original function

    state[0] = d0[3]        # co2Air
    state[1] = state[0]     # co2Top
    state[2] = 16.5         # tAir
    state[3] = state[2]     # tTop
    state[4] = state[2] + 4 # tCan
    state[5] = state[2]     # tCovIn
    state[6] = state[2]     # tCovE
    state[7] = state[2]     # tThScr
    state[8] = state[2]     # tFlr
    state[9] = state[2]     # tPipe
    state[10] = state[2]    # tSoil1
    state[11] = 0.25 * (3. * state[2] + d0[6])      # tSoil2
    state[12] = 0.25 * (2. * state[2] + 2 * d0[6])  # tSoil3
    state[13] = 0.25 * (state[2] + 3 * d0[6])       # tSoil4
    state[14] = d0[6]                               # tSoil5
    state[15] = rhMax / 100. * satVp(state[2])  # vpAir
    state[16] = state[15]   # vpTop
    state[17] = state[2]    # tLamp
    state[18] = state[2]    # tIntLamp
    state[19] = state[2]    # tGroPipe
    state[20] = state[2]    # tBlScr
    state[21] = state[4]    # tCan24
    state[22] = 0.          # cBuf
    state[23] = 9.5283e4    # cLeaf
    state[24] = 2.5107e5    # cStem
    state[25] = 5.5338e4    # cFruit
    state[26] = 3.0978e3    # tCanSum
    state[27] = time_in_days  # time

    return state

def _read_contiguous_weather(
    weather_data_dir: str,
    location: str,
    growth_year: int,
    required_end_time: float,
) -> tuple[pd.DataFrame, np.ndarray]:
    frames: list[pd.DataFrame] = []
    absolute_times: list[np.ndarray] = []
    year = growth_year
    year_offset = 0.0
    common_source_dt: float | None = None

    while True:
        path = join(weather_data_dir, location, str(year)) + ".csv"
        frame = pd.read_csv(path, sep=",")
        source_time = frame["time"].to_numpy(dtype=np.float64)
        if len(source_time) < 2 or np.any(np.diff(source_time) <= 0):
            raise ValueError(f"Weather timestamps must increase strictly: {path}")
        if not np.isclose(source_time[0], 0.0, rtol=0.0, atol=1e-6):
            raise ValueError(f"Weather source must begin at zero: {path}")

        intervals = np.diff(source_time)
        source_dt = float(np.median(intervals))
        if not np.allclose(intervals, source_dt, rtol=1e-9, atol=1e-6):
            raise ValueError(f"Weather source must have regular cadence: {path}")
        if common_source_dt is None:
            common_source_dt = source_dt
        elif not np.isclose(source_dt, common_source_dt, rtol=1e-9, atol=1e-6):
            raise ValueError(
                "Weather source cadence must remain constant across calendar "
                f"years: {path}"
            )

        shifted_time = source_time + year_offset
        frames.append(frame)
        absolute_times.append(shifted_time)
        coverage_end = float(shifted_time[-1] + source_dt)
        if coverage_end >= required_end_time:
            break

        year_seconds = (366 if calendar.isleap(year) else 365) * 86_400.0
        expected_year_end = year_offset + year_seconds
        if not np.isclose(coverage_end, expected_year_end, rtol=0.0, atol=1e-6):
            raise ValueError(
                f"Weather source does not cover the complete calendar year: {path}"
            )
        year_offset = expected_year_end
        year += 1

    return pd.concat(frames, ignore_index=True), np.concatenate(absolute_times)


def _conservative_interval_mean(
    source_time: np.ndarray,
    source_values: np.ndarray,
    target_edges: np.ndarray,
) -> np.ndarray:
    final_dt = float(np.median(np.diff(source_time)))
    source_edges = np.append(source_time, source_time[-1] + final_dt)
    if target_edges[0] < source_edges[0] or target_edges[-1] > source_edges[-1]:
        raise ValueError("Weather source does not cover the requested target grid")

    cumulative = np.concatenate(
        ([0.0], np.cumsum(source_values * np.diff(source_edges), dtype=np.float64))
    )
    target_cumulative = np.interp(target_edges, source_edges, cumulative)
    return np.diff(target_cumulative) / np.diff(target_edges)


def _calendar_daily_light_sum(
    source_time: np.ndarray,
    source_radiation: np.ndarray,
    target_time: np.ndarray,
) -> np.ndarray:
    source_dt = float(np.median(np.diff(source_time)))
    source_edges = np.append(source_time, source_time[-1] + source_dt)
    cumulative = np.concatenate(
        ([0.0], np.cumsum(source_radiation * np.diff(source_edges), dtype=np.float64))
    )
    day_start = np.floor(target_time / 86_400.0) * 86_400.0
    day_end = day_start + 86_400.0
    if day_start.min() < source_edges[0] or day_end.max() > source_edges[-1]:
        raise ValueError("Weather source does not cover complete target calendar days")

    unique_days, inverse = np.unique(day_start, return_inverse=True)
    day_energy = (
        np.interp(unique_days + 86_400.0, source_edges, cumulative)
        - np.interp(unique_days, source_edges, cumulative)
    )
    result = day_energy[inverse] * 1e-6
    return result


def computeIsDayV2(rad: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    binary = (np.asarray(rad, dtype=np.float64) > 0.0).astype(np.float64)
    linear = binary.copy()
    smooth = binary.copy()
    if len(binary) < 2:
        return linear, smooth

    transition_samples = max(1, int(np.ceil(3_600.0 / float(dt))))
    for change in np.flatnonzero(binary[1:] != binary[:-1]) + 1:
        start = max(0, int(change) - transition_samples // 2)
        end = min(len(binary), start + transition_samples)
        count = end - start
        if count <= 0:
            continue
        progress = np.linspace(0.0, 1.0, count + 2, dtype=np.float64)[1:-1]
        sigmoid = 1.0 / (1.0 + np.exp(-10.0 * (progress - 0.5)))
        sigmoid = (sigmoid - sigmoid[0]) / max(sigmoid[-1] - sigmoid[0], 1e-12)
        if binary[change] > binary[change - 1]:
            linear[start:end] = progress
            smooth[start:end] = sigmoid if count > 1 else 0.5
        else:
            linear[start:end] = 1.0 - progress
            smooth[start:end] = 1.0 - sigmoid if count > 1 else 0.5
    return linear, smooth


def load_weather_data_v2(
                        weather_data_dir: str,
                        location: str,
                        growth_year: int,
                        start_day: int,
                        n_days: int,
                        pred_horizon: int,
                        h: float,
                        nd: int,
                    ) -> np.ndarray:
    """Load weather on an exact half-open solver grid.

    Radiation is treated as an interval mean and conserves energy during
    resampling. Meteorological state variables are linearly interpolated at
    solver timestamps and bounded where their physical schema requires it.
    """
    seconds_per_day = 86_400.0
    target_dt = float(h)
    total_seconds = (float(n_days) + float(pred_horizon)) * seconds_per_day
    if target_dt <= 0 or total_seconds <= 0:
        raise ValueError("Weather duration and target timestep must be positive")
    DisturbanceSchemaV2.for_count(nd)

    sample_count = int(np.ceil(total_seconds / target_dt))
    start_time = float(start_day) * seconds_per_day
    target_edges = start_time + np.arange(sample_count + 1, dtype=np.float64) * target_dt
    target_time = target_edges[:-1]
    raw_weather, source_time = _read_contiguous_weather(
        weather_data_dir,
        location,
        growth_year,
        required_end_time=float(target_edges[-1]),
    )

    def interpolate_state(column: str) -> np.ndarray:
        values = raw_weather[column].to_numpy(dtype=np.float64)
        return np.interp(target_time, source_time, values)

    source_radiation = np.maximum(
        raw_weather["global radiation"].to_numpy(dtype=np.float64),
        0.0,
    )
    radiation = _conservative_interval_mean(
        source_time,
        source_radiation,
        target_edges,
    )
    temperature = interpolate_state("air temperature")
    relative_humidity = np.clip(interpolate_state("RH"), 0.0, 100.0)
    wind = np.maximum(interpolate_state("wind speed"), 0.0)
    sky_temperature = interpolate_state("sky temperature")

    weather_data = np.zeros((sample_count, nd), dtype=np.float64)
    weather_data[:, 0] = radiation
    weather_data[:, 1] = temperature
    vapor_density = rh2vaporDens(temperature, relative_humidity)
    weather_data[:, 2] = vaporDens2pres(temperature, vapor_density)
    weather_data[:, 3] = co2ppm2dens(temperature, 400.0) * 1e6
    weather_data[:, 4] = wind
    weather_data[:, 5] = sky_temperature
    weather_data[:, 6] = soilTempCalendar(target_time, growth_year)
    weather_data[:, 7] = _calendar_daily_light_sum(
        source_time,
        source_radiation,
        target_time,
    )
    weather_data[:, 8], weather_data[:, 9] = computeIsDayV2(radiation, target_dt)
    if nd == 11:
        precipitation_column = "precipitation accumulation"
        if precipitation_column not in raw_weather.columns:
            raise ValueError(
                "DisturbanceSchemaV2 precipitation requires source column "
                f"'{precipitation_column}'"
            )
        source_accumulation = raw_weather[precipitation_column].to_numpy(
            dtype=np.float64
        )
        if (
            not np.all(np.isfinite(source_accumulation))
            or np.any(source_accumulation < 0.0)
        ):
            raise ValueError(
                "precipitation accumulation must contain finite, non-negative values"
            )
        source_dt_hours = float(np.median(np.diff(source_time))) / 3_600.0
        source_rate = source_accumulation / source_dt_hours
        weather_data[:, 10] = _conservative_interval_mean(
            source_time,
            source_rate,
            target_edges,
        )
    return weather_data


def load_weather_data(
                        weather_data_dir: str,
                        location: str,
                        growth_year: int,
                        start_day: int,
                        n_days: int,
                        pred_horizon: int,
                        h: float,
                        nd: int,
                    ) -> np.ndarray:
    """
    Loads in raw_weather data from matlab file and converts it to values GreenLight model uses in numpy array.
    If the solver requires data on a higher frequency we interpolate between available weather data.
    Time interval of matlab data usually is 5 minutes.
    The raw_weather data is a file with 9 columns, which we convert to 7 columns used by the GreenLight.

    Args:
        weather_dataDir  - path to raw weather data
        location        - location of the weather data
        growthYear      - start of the growth year of the simulation
        startDay        - at which day of the year do we start the simulation
        nDays           - how many days do we simulate forward in time
        predHorizon     - prediction horizon [days]
        dt               - sample time of the solver in seconds
        nd              - number of weather variables
    
    Returns:
        Matrix with following interpolated weather variables:
        d[0]: iGlob         Global radiation [W m^{-2}]
        d[1]: tOut          Outdoor temperature [deg C]    
        d[2]: vpOut         Outdoor vapor pressure [Pa]
        d[3]: co2Out        Outdoor CO2 concentration [mg m^{-3}]
        d[4]: wind          Outdoor wind speed [m s^{-1}]
        d[5]: tSky          Sky temperature [deg C]
        d[6]: tSoOut        Outdoor soil temperature [deg C]
        d[7]: dli           Daily radiation sum [MJ m^{-2} day^{-1}]
        d[8]: isDay         Whether it is day or night [0,1]
        d[9]: isDaySmooth   Whether it is day or night [0,1] with a smooth transition
    """
    weather_data_path = join(weather_data_dir, location, str(growth_year)) + ".csv"

    c = 86400      # seconds in a day
    CO2_PPM = 400  # assumed constant outdoor co2 concentration [ppm]
    raw_weather = pd.read_csv(weather_data_path, sep=",")

    time = raw_weather["time"].values    # time since start of the year in [s]
    dt = np.mean(np.diff(time-time[0])) # sample period of data [s]
    N0 = int(np.ceil(start_day*c/dt))    # Start index
    Ns = int(np.ceil(n_days*c/dt))       # Number of samples we need from regular data
    Np = int(np.ceil(pred_horizon*c/dt))+1 # Number of samples into the future we need from regular data

    # check whether we exceed data length and we are in the final season
    if N0+Ns+Np > len(time):
        raw_weather = expand_weather_data(weather_data_dir, raw_weather, location, growth_year, time, dt)
    weather_data = np.zeros((Ns+Np, nd))                                         # preallocate weather data matrix
    time = raw_weather["time"].values[N0:N0+Ns+Np]                               # time since start of the year in [s]
    weather_data[:, 0] = raw_weather["global radiation"][N0:N0+Ns+Np]             # iGlob
    weather_data[:, 1] = raw_weather["air temperature"][N0:N0+Ns+Np]              # tOut
    vpDensity = rh2vaporDens(weather_data[:, 1], raw_weather["RH"][N0:N0+Ns+Np])  # vp Density
    weather_data[:,2] = vaporDens2pres(weather_data[:, 1], vpDensity)             # vpOut
    weather_data[:,3] = co2ppm2dens(weather_data[:, 1], CO2_PPM)*1e6              # co2Out (converted from kg/m^3 to mg/m^3)
    weather_data[:,4] = raw_weather["wind speed"][N0:N0+Ns+Np]                    # wind
    weather_data[:,5] = raw_weather["sky temperature"][N0:N0+Ns+Np]               # tSky
    weather_data[:,6] = soilTempNl(raw_weather["time"][N0:N0+Ns+Np])              # tSoOut
    weather_data[:, 7] = dailLightSum(time, weather_data[:,0], c)                 # daily sun radiation sum [MJ m^{-2} day^{-1}]
    weather_data[:, 8], weather_data[:,9] = computeisDay(weather_data[:, 0], dt)   # isDay, isDaySmooth

    # number of samples required for the solver
    ns = int((dt/h) * (Ns+Np))

    # interpolate and resample
    interpolation = PchipInterpolator(time, weather_data)
    timeRes = np.linspace(time[0], time[-1], ns)
    weather_dataResampled = interpolation(timeRes)

    # set small radiation values to zero
    weather_dataResampled[:,0][weather_dataResampled[:, 0] < 1e-10] = 0

    return weather_dataResampled


load_weather_data_legacy = load_weather_data

def expand_weather_data(
                    weather_data_dir: str, 
                    raw_weather: pd.DataFrame,
                    location: str,
                    growth_year: int,
                    time: ArrayLike,
                    dt: SupportsFloat,
                    ) -> np.ndarray:
    """
    Function that loads in the weather data for the next year and appends it to the current weather data.
    Required when the simulation exceeds the length of the current weather data.
    Args:
        weather_dataDir  - path to raw weather data
        raw_weather      - current weather data
        location        - location of the greenhouse
        growthYear      - year of the growth season
        time            - time since start of the year in [s]
        dt              - sample period of weather data [s]
    Returns:
        raw_weather      - weather data for the next year appended to the current weather data
    """
    weather_data_path = join(join(weather_data_dir, location), str(growth_year+1)) + ".csv"
    newRaw_weather = pd.read_csv(weather_data_path, sep=",")
    newRaw_weather["time"] += time[-1] + dt
    raw_weather = pd.concat([raw_weather, newRaw_weather.iloc[:, :]])
    return raw_weather

def days2date(timeInDays: float, referenceDate: str):
    """
    Function that converts the number of days since a reference date
    to a date in the format DD-MM-YYYY.
    Args:
        timeInDays      - number of days since reference date
        referenceDate   - reference date in format DD-MM-YYYY
    Returns:
        targetDatetime  - current date in format DD-MM-YYYY
    """
    referenceDatetime = datetime.strptime(referenceDate, '%d-%m-%Y')
    int_days = np.floor(timeInDays).astype(int)
    time_component = (timeInDays - int_days) * 24           # Convert decimal part to hours
    hours = time_component.astype(int)
    time_component = (time_component - hours) * 60          # Convert remaining decimal part to minutes
    minutes = time_component.astype(int)
    
    target_datetimes = [referenceDatetime + timedelta(days=int(int_day), hours=int(hour), minutes=int(minute)) for int_day, hour, minute in zip(int_days, hours, minutes)]

    return [target_datetime.strftime('%Y-%m-%d %H:%M:%S') for target_datetime in target_datetimes]

def computeisDay(rad: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Function that computes whether it is day or night based on the radiation.
    A day is defined as the period between sunrise and sunset.
    There is a tranisition period between day and night.
    To account for the twilight between day and night.
    Smooth transition is based on a sigmoid function.
    Args:
        rad     - radiation [W m^{-2}]
        dt      - sample period of the weather data [s]
    Returns:
        isDay           - whether it is day or night [0,1]
        isDaySmooth     - whether it is day or night [0,1] with a smooth transition
    """
    
    # add transition betwen day and night
    isDay = (rad > 0)*1.0
    isDaySmooth = deepcopy(isDay)
    transSize = int(3600/dt)    # length of transition period between night and day
                                # should be even. 3600/dt is the number of samples in an hour.

    trans = np.linspace(0, 1, transSize)
    transSmooth = 1/(1+np.exp(-10*(trans-0.5)))
    sunset = False  # indicates if we are during sunset

    for k in range(transSize, len(isDay) - transSize):
        if isDay[k] == 0:
            sunset = False
        if isDay[k] == 0 and isDay[k + 1] == 1:
            isDay[k - transSize // 2 : k + transSize // 2] = trans
            isDaySmooth[k - transSize // 2 : k + transSize // 2] = transSmooth
        elif isDay[k] == 1 and isDay[k + 1] == 0 and not sunset:
            isDay[k - transSize // 2: k + transSize // 2] = 1 - trans
            isDaySmooth[k - transSize // 2: k + transSize // 2] = 1 - transSmooth
            sunset = True
    return isDay, isDaySmooth

def dailLightSum(time: np.ndarray, rad: np.ndarray, c: int):
    """
    Function that computes the DLI (Daily Light Integral) from a given radiation time series.
    Args:
        time    - time since start of the year in [s]
        rad     - radiation [W m^{-2}]
        c       - seconds in a day
    Returns:
        lightSum    - DLI [MJ m^{-2} day^{-1}]
    """
    interval = time[1]-time[0] # time interval between samples [s]
    time = time/c               # convert to days

    # index of the midnight before current point
    mnBefore = 0

    # index of the midnight after current point
    mnAfter = np.where(np.diff(np.floor(time)) == 1)[0] +1
    if mnAfter.size == 0:
        mnAfter = len(time)
    else:
        mnAfter = mnAfter[0]
    lightSum =  np.zeros(len(time))

    for i in range(len(time)):
        lightSum[i] = np.sum(rad[mnBefore:mnAfter+1])

        if i == mnAfter -1:
            mnBefore = mnAfter
            mnAfter = np.where(np.diff(np.floor(time[mnBefore+2:])) == 1)[0] + mnBefore+2
            # mnAfter = len(time)
            if mnAfter.size == 0:
                mnAfter = len(time)
            else:
                mnAfter = mnAfter[0]
    return lightSum*interval*1e-6

# def sat_vp(temp):
#     return .61078*np.exp(17.2694*temp/(temp+238.3))

# def actual_vp(temp, rh):
#     return sat_vp(temp)*(rh/100)

# def hum_deficit(temp, rh):
#     return sat_vp(temp) - actual_vp(temp, rh)

def soilTempNl(time):
    # SOILTEMPNL An estimate of the soil temperature in the Netherlands in a given time of year
    # Based on Figure 3 in 
    # Jacobs, A. F. G., Heusinkveld, B. G. & Holtslag, A. A. M. 
    # Long-term record and analysis of soil temperatures and soil heat fluxes in 
    # a grassland area, The Netherlands. Agric. For. Meteorol. 151, 774�780 (2011).
    #
    # Input:
    #   time - seconds since beginning of the year [s]
    # Output:
    #   soilT - soil temperature at 1 meter depth at given time [�C]

    # Calculated based on a sin function approximating the figure in the reference
    
    # David Katzin, Wageningen University
    # david.katzin@wur.nl

    SECS_IN_YEAR = 3600*24*365
    soilT = 10+5*np.sin((2*np.pi*(time+0.625*SECS_IN_YEAR)/SECS_IN_YEAR))
    return soilT


def soilTempCalendar(time: np.ndarray, growth_year: int) -> np.ndarray:
    """Estimate soil temperature using phase within each actual calendar year."""
    elapsed = np.asarray(time, dtype=np.float64)
    result = np.empty_like(elapsed)
    assigned = np.zeros(elapsed.shape, dtype=bool)
    year = int(growth_year)
    year_start = 0.0

    while not np.all(assigned):
        year_seconds = (366 if calendar.isleap(year) else 365) * 86_400.0
        mask = (~assigned) & (elapsed >= year_start) & (
            elapsed < year_start + year_seconds
        )
        if np.any(mask):
            fraction = (elapsed[mask] - year_start) / year_seconds
            result[mask] = 10.0 + 5.0 * np.sin(2.0 * np.pi * (fraction + 0.625))
            assigned[mask] = True
        year_start += year_seconds
        year += 1

    return result

def vaporDens2pres(temp, vaporDens):
    # VAPORDENS2PRES Convert vapor density [kg{H2O} m^{-3}] to vapor pressure [Pa]
    #
    # Usage:
    #   vaporPres = vaporDens2pres(temp, vaporDens)
    # Inputs:
    #   temp        given temperatures [°C] (numeric vector)
    #   vaporDens   vapor density [kg{H2O} m^{-3}] (numeric vector)
    #   Inputs should have identical dimensions
    # Outputs:
    #   vaporPres   vapor pressure [Pa] (numeric vector)
    #
    # Calculation based on 
    #   http://www.conservationphysics.org/atmcalc/atmoclc2.pdf

    # David Katzin, Wageningen University
    # david.katzin@wur.nl
    # david.katzin1@gmail.com
    
    # parameters used in the conversion
    p = [610.78, 238.3, 17.2694, -6140.4, 273, 28.916]
        # default value is [610.78 238.3 17.2694 -6140.4 273 28.916]
    
    rh = vaporDens/rh2vaporDens(temp, 100) # relative humidity [0-1]
        
    satP = p[0]*np.exp(p[2]*temp/(temp+p[1]))
    # Saturation vapor pressure of air in given temperature [Pa]
    
    return satP*rh

def co2ppm2dens(temp, ppm):
    # CO2PPM2DENS Convert CO2 molar concetration [ppm] to density [kg m^{-3}]

    # Usage:
    #   co2Dens = co2ppm2dens(temp, ppm) 
    # Inputs:
    #   temp        given temperatures [�C] (numeric vector)
    #   ppm         CO2 concetration in air (ppm) (numeric vector)
    #   Inputs should have identical dimensions
    # Outputs:
    #   co2Dens     CO2 concentration in air [kg m^{-3}] (numeric vector)

    # Calculation based on ideal gas law pV=nRT, with pressure at 1 atm

    # David Katzin, Wageningen University
    # david.katzin@wur.nl
    # david.katzin1@gmail.com

    R = 8.3144598 # molar gas constant [J mol^{-1} K^{-1}]
    C2K = 273.15 # conversion from Celsius to Kelvin [K]
    M_CO2 = 44.01e-3 # molar mass of CO2 [kg mol^-{1}]
    P = 101325 # pressure (assumed to be 1 atm) [Pa]
    
    # number of moles n=m/M_CO2 where m is the mass [kg] and M_CO2 is the
    # molar mass [kg mol^{-1}]. So m=p*V*M_CO2*P/RT where V is 10^-6*ppm    
    return P*10**-6*ppm*M_CO2/(R*(temp+C2K))

def co2dens2ppm(temp, dens):
    '''
    Convert CO2 density [kg m^{-3}] to molar concentration [ppm]
    '''
    R = 8.3144598        # Molar gas constant [J mol^{-1} K^{-1}]
    C2K = 273.15         # Conversion from Celsius to Kelvin [K]
    M_CO2 = 44.01e-3     # Molar mass of CO2 [kg mol^{-1}]
    P = 101325           # Pressure (assumed to be 1 atm) [Pa]
    
    return 1e6 * R * (temp + C2K) * dens / (P * M_CO2)

def satVp(temp):
    # saturated vapor pressure (Pa) at temperature temp (C)
    # Calculation based on 
    #   http://www.conservationphysics.org/atmcalc/atmoclc2.pdf
    # See also file atmoclc2.pdf

    # parameters used in the conversion
    # p = [610.78 238.3 17.2694 -6140.4 273 28.916];
        # default value is [610.78 238.3 17.2694 -6140.4 273 28.916]

        # Saturation vapor pressure of air in given temperature [Pa]
    return 610.78* np.exp(17.2694*temp/(temp+238.3))


def vaporPres2rh(temp, vaporPres):
    return np.clip(100*vaporPres/satVp(temp), a_min=0., a_max=100.)

def vaporDens2rh(temp, vaporDens):
    """
    vaporDens2rh Convert vapor density [kg{H2O} m^{-3}] to relative humidity [%]

    Usage:
    rh = vaporDens2rh(temp, vaporDens)
    Inputs:
    temp        given temperatures [°C] (numeric vector)
    vaporDens   absolute humidity [kg{H20} m^{-3}] (numeric vector)
    Inputs should have identical dimensions
    Outputs:
    rh          relative humidity [%] between 0 and 100 (numeric vector)

    Calculation based on 
    http://www.conservationphysics.org/atmcalc/atmoclc2.pdf

    David Katzin, Wageningen University
    david.katzin@wur.nl
    """
    # constants
    # molar gas constant [J mol^{-1} K^{-1}]
    R = 8.3144598 
    # conversion from Celsius to Kelvin [K]
    C2K = 273.15  
    # molar mass of water [kg mol^-{1}]
    Mw = 18.01528e-3  
    
    # parameters used in the conversion
    # default value is [610.78 238.3 17.2694 -6140.4 273 28.916]
    p = [610.78, 238.3, 17.2694, -6140.4, 273, 28.916]
    
    # Saturation vapor pressure of air in given temperature [Pa]
    satP = p[0]*np.exp(p[2]*temp/(temp+p[1])) 
    # convert to relative humidity using the ideal gas law pV=nRT => n=pV/RT 
    # so n=p/RT is the number of moles in a m^3, and Mw*n=Mw*p/(R*T) is the 
    # number of kg in a m^3, where Mw is the molar mass of water.
    relhumid = 100*R*(temp+C2K)/(Mw*satP)*vaporDens
    # if np.isinf(relhumid).any():
    #     print(temp, vaporDens)
    return np.clip(relhumid, a_min=0, a_max=100)

def rh2vaporDens(temp, rh):
    # RH2VAPORDENS Convert relative humidity [#] to vapor density [kg{H2O} m^{-3}]

    # Usage:
    #   vaporDens = rh2vaporDens(temp, rh)
    # Inputs:
    #   temp        given temperatures [�C] (numeric vector)
    #   rh          relative humidity [#] between 0 and 100 (numeric vector)
    #   Inputs should have identical dimensions
    # Outputs:
    #   vaporDens   absolute humidity [kg{H20} m^{-3}] (numeric vector)

    # Calculation based on 
    #   http://www.conservationphysics.org/atmcalc/atmoclc2.pdf

    # David Katzin, Wageningen University
    # david.katzin@wur.nl
    # david.katzin1@gmail.com

    # constants
    R = 8.3144598 # molar gas constant [J mol^{-1} K^{-1}]
    C2K = 273.15 # conversion from Celsius to Kelvin [K]
    Mw = 18.01528e-3 # molar mass of water [kg mol^-{1}]
    
    # parameters used in the conversion
    p = [610.78, 238.3, 17.2694, -6140.4, 273, 28.916]
    # default value is [610.78 238.3 17.2694 -6140.4 273 28.916]
    
    satP = p[0]*np.exp(p[2]*temp/(temp+p[1]))
        # Saturation vapor pressure of air in given temperature [Pa]
    
    pascals=(rh/100)*satP # Partial pressure of vapor in air [Pa]
    
    # convert to density using the ideal gas law pV=nRT => n=pV/RT 
    # so n=p/RT is the number of moles in a m^3, and Mw*n=Mw*p/(R*T) is the 
    # number of kg in a m^3, where Mw is the molar mass of water.
    
    return pascals*Mw/(R*(temp+C2K))

def compute_sky_temp(air_temp, cloud):
    """
    Compute sky temperature from air temperature and cloud cover.
    Args
        air_temp: Air temperature in °C
        cloud: Cloud cover (0-1)
    Returns
        sky_temp: Sky temperature in °C
    """

    sigma = 5.67e-8 # Stefan-Boltzmann constant
    C2K = 273.15    # Conversion of °C to K

    ld_clear = 213+5.5*air_temp                      # Equation 5.26
    eps_clear = ld_clear/(sigma*(air_temp+C2K)**4)    # Equation 5.22
    eps_cloud = (1-0.84*cloud)*eps_clear+0.84*cloud   # Equation 5.32
    ld_cloud = eps_cloud*sigma*(air_temp+C2K)**4      # Equation 5.22
    sky_temp = (ld_cloud/sigma)**(0.25)-C2K           # Equation 5.22, but here assuming eps=1
    return sky_temp
