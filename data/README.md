# Input time-series data

These time series describe water levels along the Western Scheldt estuary, Netherlands, over **5–7 December 2013** (storm Xaver). They were supplied as input datasets for TU Delft's WI4475 *Data Assimilation — Tidal wave* project.

| Filenames | Contents |
|---|---|
| `tide_*.txt` | Astronomical/tidally filtered water levels, including Cadzand boundary forcing |
| `waterlevel_*.txt` | Measured water levels including storm effects (Vlissingen, Terneuzen, Hansweert and Bath) |

Records use `YYYYMMDDHHMM` timestamps in UTC and water-level values in metres. The file headers identify the series as retrieved from the **MATROOS** database. The course material attributes the underlying measurements to **Rijkswaterstaat**. Public Rijkswaterstaat Waterinfo measurements are listed as public-domain/CC0 ([government catalogue](https://data.overheid.nl/dataset/9442-waterinfo-rijkswateren), [official terms](https://waterinfo-extra.rws.nl/monitoring/gebruik-data/)). The tidally filtered series are course-provided extracts, which may differ from raw downloads; confirm reuse conditions of these particular processed extracts if redistribution is a concern.

The four monitoring stations used for evaluation are Vlissingen, Terneuzen, Hansweert and Bath. Locations in the numerical model are idealised along-estuary distances, not exact geodetic coordinates.
