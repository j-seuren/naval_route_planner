# Multi-objective Ship Weather Route Optimization

A route planner for ocean-going vessels optimizing for minimum travel time and/or minimum fuel cost.
The route planner uses the multi-objective algorithm NSGA-II as an evolutionary framework to obtain a set of Pareto optimal routes.

## Main features

- Multiple objectives
    - Minimum travel time
    - Minimum fuel cost
- Variable vessel speed
- Environmental conditions
    - Weather forecasts (or historical weather)
    - Historical ocean currents
- Emission control areas (areas with higher fuel cost)
- Major shipping canals
    - Panama Canal
    - Suez Canal
- Navigation on coastlines
    - Optional: avoiding shallow waters

### Input
**Required**
- Start and destination coordinates (longitude, latitude)
- Discrete set of attainable vessel speeds and corresponding fuel consumption

**Optional**
- Fuel price per metric ton
- ECA fuel price multiplication factor
- Constant speed setting
- Include weather (-> set departure date and ship characteristics: Length, Displacement, block coefficient)
- Include current (-> set departure date)

### Output

- List of waypoints in (longitude, latitude) degrees
- Fuel cost x1000 (currency depends on fuel price currency)
- Travel time in days
- Total distance in nautical miles
- Crossed canals

## Getting started

To use, do

    >>> from ortec-ship-route-planner import RoutePlanner
    >>> planner = RoutePlanner(**kwargsPlanner)
	>>> raw_output = planner.compute(startEnd, **kwargsCompute)
	>>>  processed_output, raw_output = planner.post_process(raw_output)

### Input arguments
Keyword arguments `RoutePlanner` `kwargsPlanner` are
- `constantSpeedIdx` (default `None`), _int_, index speeds list corresponding to the selected constant speed. If `None`, speed is variable.
- `vesselID` (default `Fairmaster_2`), _str_, name of the vessel
- `shipLoading` (default `normal`), _str_, ship loading. Either `normal` or `ballast`. Depends on characteristics
- `ecaFactor` (default `1.5593`) _int_ or _float_, emission control area fuel price multiplication factor
- `fuelPrice` (default `300`) : _int_ or _float_, the fuel price per metric ton
- `bathymetry` (default `False`) : _bool_, include (`True`) or exclude (`False`) shallow waters / bathymetry
- `algorithmArgs` (default `None`) : _dict_, input parameters, see below.
- `tb` (default `None`) : _class_, DEAP toolbox
- `criteria` (default `{'minimalTime': True, 'minimalCost': True}`) : _dict_, criteria to optimize
- `seed` (default `None`) : random gerator seed

	
Keyword arguments `compute` `kwargsCompute` are
- `recompute` (default `False`), _bool_, recompute routes (`True`) or load routes if previously calculated (`False`)
- `startDate` (default `None`), _datetime_, departure date (year, month, day) needed for environmental condition optimization
- `current` (default `False`), _bool_, include (`True`) or exclude (`False`) ocean currents
- `weather` (default `False`), _bool_, include (`True`) or exclude (`False`) weather
- `avoidArctic` (default `True`), _bool_, include (`True`) or exclude (`False`) Arctic Circle
- `avoidAntarctic` (default `True`), _bool_, include (`True`) or exclude (`False`) Antarctic Circle

### Returned values
`raw_output` is a raw output dictionary containing
- `startEnd`: _tuple_ of _tuple_ of pair of _float_, longitude/latidue pairs of start and destination location in degrees
- `initialRoutes`: _list_, initial routes found on graph
- `logs`: _list_, list of DEAP logbook dictionaries for each (sub)route
- `fronts`: _list_, list of DEAP ParetoFront dictionaries for each (sub)route

`processed_output` is a processed output dictionary containing
- `routeresponse` _list_ with each entry a route _dict_:
    - `optimizationCriterion`, _str_, optimization criterion of route 
    - `bestWeighted`: `False`, _bool_, indicating if the route is best weighted in Pareto front (depends on fitness value weights)
    - `distance`: _float_, distance in nautical miles
    - `fuelCost`: _float_, fuel cost x1000
    - `travelTime`: _float_, travel time in days
    - `fitValues`: _list_ of _float_, optimization fitness values. Only different to objective values `travelTime` and `fuelCost` if post_process evaluation has different settings.
    - `waypoints`: _list_ of _tuple_ of pair of _float_, each tuple contains longitude/latidue pair in degrees
    - `crossedCanals`: _list_ of _str_, crossed canals

## Dependencies

- [Dask](https://docs.dask.org/en/latest/)
- [DEAP](http://deap.readthedocs.org/)
- [Fiona](https://github.com/Toblerity/Fiona)
- [Haversine](https://github.com/mapado/haversine)
- [More Itertools](https://github.com/more-itertools/more-itertools)
- [Networkx](https://networkx.org/)
- [NumPy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
    - [xlrd](http://www.python-excel.org/)
- [pyproj](https://github.com/pyproj4/pyproj)
- [Requests](https://requests.readthedocs.io/en/master/)
- [Rtree](https://github.com/Toblerity/rtree)
- [SCOOP](https://github.com/soravux/scoop)
- [SciPy](https://www.scipy.org/)
- [Shapely](https://github.com/Toblerity/Shapely)
- [xarray](https://xarray.pydata.org/)

_Optional_

- [matplotlib](https://matplotlib.org/)
 - [basemap](https://matplotlib.org/basemap)

_Web_application_

- flask
- flask_cors

## Environmental conditions
### Weather (wind)

The planner may use historic or forecasted weather data for weather optimization.
By default it uses forecasted weather data with a maximum of 184 hour forecast time horizon.
In order to get forecasted weather data, the planner uses GRIB files download using the procedure described [here](https://www.cpc.ncep.noaa.gov/products/wesley/get_gfs.html).

### Ocean current

Affects vessel speed and course.
Uses historical (satellite-derived) surface geostrophic and Ekman current data obtained from [GlobCurrent.org](http://www.globcurrent.org/).
Time period: 1993 to 2016

## Default input parameters
**Navigation area parameters**
- `avoidAntarctic`: `True`,
- `avoidArctic`: `True`,
- `res`: `i`,                Resolution of shorelines
- `penaltyValue`: `1`,         Penalty value for Bathymetry
- `recursionLvl`: `4`,         Main recursion level graph
- `varRecursionLvl`: `6`,      Variable recursion level graph
- `splits`: `3`,               Threshold for split_polygon (3 yields best performance)

**MOEA parameters**
- `popSize`: `336`,            Population size
- `nBar`: `100`,              Local archive size
- `crossoverProb`: `0.81`,    Crossover probability
- `mutationProb`: `0.28`,     Mutation probability
- `maxMoves`: `9`,            Max. number of mutations per selected individual

**Termination parameters**
- `maxEvaluations`: `None`,
- `minGenerations`: `150`,      Minimal number of generations
- `maxGDs`: `40`,               Max length of generational distance list
- `minVarianceGD`: `1e-6`,      Minimal variance of generational distance list

**Mutation parameters**
- `mutationOperators`: `['speed', 'insert', 'move', 'delete']`,   Operators to be included
- `widthRatio`: `4.22`,         Shape width ratio for waypoint insertion
- `radius`: `1.35`,             Circle radius for waypoint translation
- `scaleFactor`: `0.1`,         Exponential distribution scale factor for # edges selection
- `delFactor`: `1.2`,           Multiplication factor for 'delete move' selection weight
- `gauss`: `False`,             Use Gaussian mutation for insert and move operators

**Evaluation parameters**
- `segLengthF`: `15`,           Length of linear approx. of great circle track for feasibility
- `segLengthC`: `8`             Length of linear approx. of great circle track for ocean currents and wind data point selection

## Applications

### Route visualization web application

Web application for visualization of extreme routes in Pareto front (minimal time, minimal fuel cost).
See set-up guide for installation instructions.

![Application screenshot](https://github.com/j-seuren/naval_route_planner/blob/images/Route_visualization_app_1.jpg?raw=true)
![Example voyage](https://github.com/j-seuren/naval_route_planner/blob/images/Route_visualization_app_2.jpg?raw=true)

### Project location distances

Example application of computing minimal distance routes between multiple locations accross the globe.

## Recommendations for further improvement

### Computation time
- Started with multiprocessing: No speed improvement yet. Probably due to lack of caching intermediate results (distance, speed, etc.) between processes.
- Delete as much waypoints as possible from initial routes
- Clip/shrink sailing region (reducing ocean current / weather data in memory)
- For minimizing distance:
    - Remove ‘Change speed’ operator, as speed has no effect
    - Minimize single objective (travel time)
- Travel time constraints (penalize routes >𝑥 days
    - e.g. for time windows at canals and ports
- Discard routes avoiding canals with 𝑥% longer travel time
- Tune model parameters

### Solution quality
- Validate vessel speed reduction due to weather (wind)
- Obtain/create bathymetry data for water depth of ~20m
    - Include tides for coastal navigation (e.g. Waddenzee)?
- Add route mutations
    - Split waypoint into two waypoints
    - Merge two waypoints into one
- Extra Include arctic routing / seasonality ice
- Extra high-cost areas include in (fuel) cost or 3rd objective:
    - Piracy risk areas
    - Seasonal heavy weather areas

_NB: additional high-cost areas can be included as polygons in the same way as ECAs/shallow waters are included._