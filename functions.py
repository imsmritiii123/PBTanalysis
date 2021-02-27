import math
import numpy as np


def generate_distance(lat1, lon1, lat2, lon2):
    """Calculate distace between two points

    Args:
        lat1 (float): latitude of first point
        lon1 (float): longitude of first point
        lat2 (float): latitude of second point
        lon2 (float): longitude of second point

    Returns:
        float: distance in meters
    """
    R = 6371000
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (math.sin(math.pi/180 * dlat/2))**2 + math.cos(math.pi / 180 * lat1) * \
        math.cos(math.pi/180 * lat2) * (math.sin(math.pi/180*dlon/2))**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = R * c
    return d


def distances(lons1, lats1, lons2, lats2):
    """Calculates distance between two array of points

    Args:
        lons1 (array): Array of first longitudes
        lats1 (array): Array of first latitudes
        lons2 (array): Array of second longitudes
        lats2 (array): Array of second latitudes

    Returns:
        array: Array of distances between two array of points
    """
    R = 6371000
    dlon = lons2 - lons1
    dlat = lats2 - lats1
    a = (np.sin(np.pi/180 * dlat/2))**2 + np.cos(np.pi / 180 * lats1) * \
        np.cos(np.pi/180 * lats2) * (np.sin(np.pi/180*dlon/2))**2
    c = 2 * np.arctan(np.sqrt(a), np.sqrt(1-a))
    d = R * c
    return d


def minimum_distance(lat, lon, lats2, lons2):
    """Calculate minimum distance between point and array of points

    Args:
        lat (float): Latitude of a point
        lon (float): Longitude of a point
        lats2 (array): Array of latitudes
        lons2 (array): Array of longitudes

    Returns:
        float: minimum distance
    """
    R = 6371000
    l = np.ones(len(lats2))
    lats1 = l*lat
    lons1 = l*lon
    dlon = lons2 - lons1
    dlat = lats2 - lats1
    a = (np.sin(np.pi/180 * dlat/2))**2 + np.cos(np.pi / 180 * lats1) * \
        np.cos(np.pi/180 * lats2) * (np.sin(np.pi/180*dlon/2))**2
    c = 2 * np.arctan(np.sqrt(a), np.sqrt(1-a))
    d = R * c
    i = np.where(d == min(d))[0][0]
    return [transit_station_names[i], min(d)]


def minimum_distances(lats, lons):
    distances = []
    names = []
    for i in range(len(lats)):
        [name, d] = minimum_distance(lats[i], lons[i], np.array(
            transit_station_lats), np.array(transit_station_lons))
        distances.append(d)
        names.append(name)
    return [distances, names]


def generate_bus_df(bus_id):
    """Generate dataframe for given bus_id

    Args:
        bus_id (int): Bus ID

    Returns:
        df: Dataframe for give bus
    """
    df = df_full[df_full['bus_id'] == bus_id]
    df.drop_duplicates(subset='last_update', inplace=True)
    return df


def plot_on_map(df, BB, map_, s=10, alpha=0.1):
    fig, axs = plt.subplots(figsize=(200*(BB[1]-BB[0]), 200*(BB[3]-BB[2])))
    axs.scatter(df.longitude, df.latitude, zorder=1, alpha=alpha, c='r', s=s)
    axs.set_xlim((BB[0], BB[1]))
    axs.set_ylim((BB[2], BB[3]))
    axs.set_title('GPS Locations')
    axs.imshow(map_, zorder=0, extent=BB)


def generate_bus_day_df(bus_id, day, plot=False, map_=False):
    df_temp = generate_bus_df(bus_id)
    df_temp = df_temp[df_temp['datetime'].dt.dayofyear == day]
    ax = sns.scatterplot(data=df_temp, x='longitude', y='latitude')
    ax_d = plt.gca()

    if not plot:
        plt.close()

    if map_:
        route_map = plt.imread('./Bus Routes/Lagankhel-NayaBusPark.png')
        BB = [ax_d.get_xlim()[0], ax_d.get_xlim()[1],
              ax_d.get_ylim()[0], ax_d.get_ylim()[1]]
        plot_on_map(df_temp, BB, route_map)

    return df_temp


def generate_movement_days_for_bus(bus_df, distance=5000):
    """Generate days where the bus travelled more than given distance

    Args:
        bus_id (int): Bus ID
        distance (int, optional): Distance in meters. Defaults to 5000.

    Returns:
        list: list of days
    """
    bus_df = bus_df.drop_duplicates(subset='last_update')
    bus_days = bus_df['datetime'].dt.dayofyear.unique()
    movement_dfs = []
    movement_days = []
    for day in bus_days:
        bus_day_df = bus_df[bus_df['datetime'].dt.dayofyear == day]
        if bus_day_df.shape[0] > 50:
            lat1 = bus_day_df['latitude'].max()
            lat2 = bus_day_df['latitude'].min()
            lon1 = bus_day_df['longitude'].max()
            lon2 = bus_day_df['longitude'].min()
            if generate_distance(lat1, lon1, lat2, lon2) > distance:
                movement_dfs.append(bus_day_df)
                movement_days.append(day)
    return [movement_dfs, movement_days]


def plot_movement(bus):
    bus_df = generate_bus_df(bus)
    m_days = generate_movement_days_for_bus(bus)
    fig, axs = plt.subplots(math.ceil(
        len(m_days)/3), 3, figsize=(16, 4*math.ceil(len(m_days)/3)), squeeze=False)
    for i in range(len(m_days)):
        df_temp = bus_df[bus_df['datetime'].dt.dayofyear == m_days[i]]
        sns.scatterplot(data=df_temp, x='longitude', y='latitude',
                        ax=axs[i//3][i % 3], label=m_days[i])
    plt.show()


def animate_f(df):
    print('For {} datapoints'.format(df.shape[0]))
    df.drop_duplicates(subset='last_update', inplace=True)
    print('For {} datapoints'.format(df.shape[0]))
    x_data = df['longitude'].values
    y_data = df['latitude'].values

    plt.plot(x_data, y_data, 'o')
    ax_d = plt.gca()
    plt.close()

    fig, ax = plt.subplots()
    ax.set_xlim(ax_d.get_xlim())
    ax.set_ylim(ax_d.get_ylim())

    lines = plt.plot([], 'o')
    line = lines[0]

    def animate(frame):
        x = x_data[:frame]
        y = y_data[:frame]
        line.set_data((x, y))

    anim = FuncAnimation(fig, func=animate, frames=len(x_data), interval=100)
    video = anim.to_html5_video()
    html = display.HTML(video)
    display.display(html)
    plt.close()


def generate_movement_days(raw_df):
    """Generates the movement days for all buses in df_full

    Returns:
        dict: Bus_ID: movement days
    """
    buses = raw_df['bus_id'].unique()
    m_days = {}
    m_dfs = {}
    for bus in buses:
        bus_df = raw_df[raw_df['bus_id'] == bus]
        print('Calculating movement days for bus {}'.format(bus))
        [m_dfs[bus], m_days[bus]] = generate_movement_days_for_bus(bus_df)
    return [m_dfs, m_days]


def generate_nearest_station(lon, lat, stations):
    """Generate nearst station for the given point from the dataframe of all stations in the route

    Args:
        lon (float): longitude of point
        lat (float): latitude of point
        stations (df): DataFrame of stations in the route

    Returns:
        list: Minimum distance between the point and the nearest station, Longitude of nearest station, Latitude of nearest station
    """
    R = 6371000
    lats2 = stations['latitude'].values
    lons2 = stations['longitude'].values
    l = np.ones(len(lats2))
    lats1 = l*lat
    lons1 = l*lon
    dlon = lons2 - lons1
    dlat = lats2 - lats1
    a = (np.sin(np.pi/180 * dlat/2))**2 + np.cos(np.pi / 180 * lats1) * \
        np.cos(np.pi/180 * lats2) * (np.sin(np.pi/180*dlon/2))**2
    c = 2 * np.arctan(np.sqrt(a), np.sqrt(1-a))
    d = R * c
    i = np.where(d == min(d))[0][0]
    nearest_station = stations.iloc[i]
    return [min(d), nearest_station['name'], [nearest_station['longitude'], nearest_station['latitude']]]


def append_nearest_stations(route, stations, maxDistance=150, drop_duplicates=True):
    """Generate nearest stations for the points in the roue

    Args:
        route (df): DataFrame of the points in route
        stations (df): DataFrame of the stations of the route
        maxDistance (int, optional): Distance for the point in route to be considered near the station. Defaults to 100 meters.
        drop_duplicates (bool, optional): Check to drop duplicates of bus stations. Defaults to True.

    Returns:
        df: Returns the dataframe with info of the nearest station appended to it.
    """
    distance_to_nearest_station = []
    nearest_station_name = []
    latitude_of_nearest_station = []
    longitude_of_nearest_station = []
    route.sort_values('last_update', inplace=True)
    lons = route['longitude'].values
    lats = route['latitude'].values
    for i in range(route.shape[0]):
        [d, name, [longitude, latitude]] = generate_nearest_station(
            lons[i], lats[i], stations)
        distance_to_nearest_station.append(d)
        nearest_station_name.append(name)
        longitude_of_nearest_station.append(longitude)
        latitude_of_nearest_station.append(latitude)

    route['name'] = nearest_station_name
    route['nlongitude'] = longitude_of_nearest_station
    route['nlatitude'] = latitude_of_nearest_station
    route['distance'] = distance_to_nearest_station

    if drop_duplicates:
        route = route[route['distance'] < maxDistance]
        return route.drop_duplicates(subset='name')
    else:
        return route


def parse_kml(filename):
    """Parses the kml file

    Args:
        filename (string): The full file name along with the location of the file

    Returns:
        dict: route name, route description, route stops, forward route and backward route
    """

    route = {}
    with open(filename, "r") as content:
        soup = BeautifulSoup(content, "lxml-xml")
    route['name'] = soup.find('name').text
    route['full_name'] = soup.find('description').text
    folders = soup.findAll('Folder')

    route_coordinates = []

    for folder in folders:
        folder_name = folder.find('name').text
        if folder_name == 'Bus Stops':
            stop_names = []
            stop_coordinates = []
            for point in folder.findAll('Placemark'):
                stop_name = point.find('name').text
                coordinate = point.find('coordinates').text
                stop_names.append(stop_name)
                stop_coordinates.append(coordinate)
            stop_coordinates = [coordinate.split(
                ',')[:-1] for coordinate in stop_coordinates]
            stop_lons = np.array([float(coordinate[0])
                                  for coordinate in stop_coordinates])
            stop_lats = np.array([float(coordinate[1])
                                  for coordinate in stop_coordinates])
            route['stops'] = pd.DataFrame(
                {'name': stop_names, 'longitude': stop_lons, 'latitude': stop_lats})

        else:
            route_coordinates = folder.find(
                'LineString').coordinates.text.split('\n')
            route_coordinates = [coordinate.strip()
                                 for coordinate in route_coordinates][1:-1]
            route_coordinates = [coordinate.split(
                ',')[:-1] for coordinate in route_coordinates]
            route_lons = np.array([float(coordinate[0])
                                   for coordinate in route_coordinates])
            route_lats = np.array([float(coordinate[1])
                                   for coordinate in route_coordinates])
            route[folder_name] = pd.DataFrame(
                {'longitude': route_lons, 'latitude': route_lats})
    return route


def append_endpoint(route_df, endpoint0, endpoint1, cropToEndpoints=True):
    """Appends True if stop is endpoint for the given route

    Args:
        route_df (df): DataFrame of given route
        endpoint0 (str): First endpoint
        endpoint1 (str): Second point
        cropToEndpoints (bool, optional): If true crops to endpoint. Defaults to True.

    Returns:
        df: DataFrame with appended True/False depending if it is endpoint
    """
    print(route_df.shape)
    route_df.sort_values(['last_update'], inplace=True)
    route_df['Endpoint0'] = route_df['name'] == endpoint0
    route_df['Endpoint1'] = route_df['name'] == endpoint1
    route_df['Endpoint'] = np.logical_or(
        route_df['Endpoint0'], route_df['Endpoint1'])
    route_df.drop(columns=['Endpoint0', 'Endpoint1'], inplace=True)
    if route_df['Endpoint'].sum() < 2:
        return route_df.iloc[0]
    if cropToEndpoints:
        start = route_df[route_df['Endpoint'] == True].index[0]
        end = route_df[route_df['Endpoint'] == True].index[-1]
        route_df = route_df.loc[start:end]
    route_df.reset_index(inplace=True)
    route_df.drop(columns=['index'], inplace=True)

    return route_df


def generate_trip_endpoints(route_df):
    trip_endpoints = np.where(np.logical_xor(
        route_df['Endpoint'][:-1].values, route_df['Endpoint'][1:].values) == True)[0]

    trip_endpoints[1::2] += 1
    names = route_df.iloc[trip_endpoints]['name'].values
    indices = []
    for i in range(len(names)):
        if i == 0:
            if names[i] != names[i+1]:
                indices.append(i)
        elif i == len(names)-1:
            if names[i] != names[i-1]:
                indices.append(i)
        else:
            if (names[i] != names[i-1] or names[i] != names[i+1]):
                indices.append(i)
    trip_endpoints = trip_endpoints[indices]

    return trip_endpoints


def generate_trips(route_df, endpoint0, endpoint1):
    """Generate trips for given route_df when given two endpoints

    Args:
        route_df (df): DataFrame of route
        endpoint0 (str): First endpoint
        endpoint1 (str): Second endpoint

    Returns:
        list: List of trips
    """
    trips = []
    route_with_endpoint_df = append_endpoint(route_df, endpoint0, endpoint1)
    if route_with_endpoint_df.shape[0] < 20:
        return trips
    trip_endpoints = generate_trip_endpoints(route_with_endpoint_df)

    for i in range(0, len(trip_endpoints), 2):
        trip = route_with_endpoint_df.iloc[trip_endpoints[i]: trip_endpoints[i+1]+1]
        if trip.shape[0] > 20:
            trips.append(trip)
    return trips


def generate_training_data(df_tmp):
    df_tmp_bus_stops = df_tmp.iloc[1:-1]['distance'] < 100
    df_tmp_bus_stops = df_tmp_bus_stops.values
    df_tmp_bus_stops = np.concatenate(([True], df_tmp_bus_stops, [True]))
    df_tmp = df_tmp[df_tmp_bus_stops]
    names = df_tmp['name'].values
    indices = [0]
    for i in range(1, len(names)-1):
        if (names[i] != names[i-1] or names[i] != names[i+1]):
            indices.append(i)
    indices.append(i+1)
    df_tmp = df_tmp.iloc[indices]

    names_from = df_tmp['name'][:-1].values
    names_to = df_tmp['name'][1:].values
    time_from = df_tmp['last_update'][:-1].values
    time_to = df_tmp['last_update'][1:].values

    lons1 = df_tmp['longitude'][:-1].values
    lats1 = df_tmp['latitude'][:-1].values
    lons2 = df_tmp['longitude'][1:].values
    lats2 = df_tmp['latitude'][1:].values
    d = distances(lons1, lats1, lons2, lats2)

    df = pd.DataFrame({'From': names_from, 'To': names_to,
                       'Start': time_from, 'End': time_to, 'Distance': d})
    df['Time_Seconds'] = (df['End'] - df['Start']).dt.seconds

    return df


def generate_training_data_alt(trip_df):
    """Generates training suitable data from the given trip

    Args:
        trip_df (df): DataFrame of trip

    Returns:
        df: DataFrame of training suitable data
    """
    DISTANCE = 200
    trip_df_bus_stops = trip_df.iloc[1:-1]['distance'] < DISTANCE
    trip_df_bus_stops = trip_df_bus_stops.values
    # First and last must be stops in the trip
    trip_df_bus_stops = np.concatenate(([True], trip_df_bus_stops, [True]))
    trip_df = trip_df[trip_df_bus_stops]
    names = trip_df['name'].values
    indices = [0]
    for i in range(1, len(names)-1):
        if (names[i] != names[i-1] or names[i] != names[i+1]):
            indices.append(i)
    indices.append(i+1)
    trip_df = trip_df.iloc[indices]
    names = trip_df['name'].values
    times = trip_df['last_update'].values

    dfs = []

    for i in range(len(names)-1):
        from_ = pd.DataFrame({'from': names[i:i+1], 'Start': times[i:i+1]})
        l = len(names)-1-i
        end = len(names)

        df = pd.concat([from_]*l)
        df['to'] = names[i+1:end]
        df['End'] = times[i+1:end]
        df.drop_duplicates(subset=['from', 'to'], inplace=True)
        if df.iloc[0]['from'] == df.iloc[0]['to']:
            continue
        dfs.append(df)
    return pd.concat(dfs)


def generate_all_training_data(df_full, buses):
    """Generate all training data from df_full

    Returns:
        df: training suitable df
    """

    [movements, m_days] = generate_movement_days(df_full)
    dfs_outer = []
    bus_ids = []
    for bus in movements:
        dfs_inner = []
        kml_file = 'Bus Routes/' + buses.loc[bus]['route_short'] + '.kml'
        stations_in_route = parse_kml(kml_file)['stops']
        for df, m_day in zip(movements[bus], m_days[bus]):
            df = append_nearest_stations(
                df, stations_in_route, drop_duplicates=False)
            dfs = []
            trips = generate_trips(
                df, stations_in_route['name'].iloc[0], stations_in_route['name'].iloc[-1])
            print('Bus: {} - Day: {} - Tours: {}  '.format(bus, m_day, len(trips)))
            for trip in trips:
                if trip.shape[0] > 10:
                    df = generate_training_data_alt(trip)
                    dfs.append(df)
            if len(dfs) < 1:
                continue
            df_inner = pd.concat(dfs)
            dfs_inner.append(df_inner)
        if len(dfs_inner) > 0:
            df_outer = pd.concat(dfs_inner)
            dfs_outer.append(df_outer)

            bus_id = [bus] * df_outer.shape[0]
            bus_ids.append(bus_id)

    train = pd.concat(dfs_outer)
    train['Segment'] = train['from'] + ' - ' + train['to']
    train.set_index('Segment', inplace=True)
    train['Year'] = train['Start'].dt.year
    train['Month'] = train['Start'].dt.month
    train['Day'] = train['Start'].dt.day
    train['Hour'] = train['Start'].dt.hour
    train['WeekDay'] = train['Start'].dt.strftime('%a')
    train['Duration'] = (train['End'] - train['Start']).dt.seconds
    train.drop(columns=['from', 'to', 'End'], inplace=True)
    return train
