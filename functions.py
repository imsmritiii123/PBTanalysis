def bus_df(bus_id):
    return df_full[df_full['bus_id']==bus_id]

def plot_on_map(df, BB, map_, s=10, alpha=0.1):
    fig, axs = plt.subplots(figsize=(200*(BB[1]-BB[0]), 200*(BB[3]-BB[2])))
    axs.scatter(df.longitude, df.latitude, zorder=1, alpha=alpha, c='r', s=s)
    axs.set_xlim((BB[0], BB[1]))
    axs.set_ylim((BB[2], BB[3]))
    axs.set_title('GPS Locations')
    axs.imshow(map_, zorder=0, extent=BB)

def bus_day_df(bus_id, day, plot=False, map_=False):
    df_temp = bus_df(bus_id)
    df_temp = df_temp[df_temp['datetime'].dt.dayofyear == day]
    ax = sns.scatterplot(data=df_temp, x='longitude', y='latitude')
    ax_d = plt.gca()
    
    if not plot:
        plt.close()

    if map_:
        route_map = plt.imread('./Bus Routes/Lagankhel-NayaBusPark.png')
        BB = [ax_d.get_xlim()[0], ax_d.get_xlim()[1], ax_d.get_ylim()[0], ax_d.get_ylim()[1]]
        plot_on_map(df_temp, BB, route_map)

    return df_temp

def movement_days(bus):
    df_temp = bus_df(bus)
    days = df_temp['datetime'].dt.dayofyear.unique()
    movement_days = []
    for day in days:
        df_temp = bus_day_df(bus, day)
        df_temp_del_lat = df_temp['latitude'].max() - df_temp['latitude'].min()
        df_temp_del_lon = df_temp['longitude'].max() - df_temp['longitude'].min()
        df_temp_del = min(df_temp_del_lat*1000, df_temp_del_lon*1000)
        if (df_temp.shape[0]>100 and df_temp_del>1.5):
            movement_days.append(day)
    return movement_days

def plot_movement(bus):
    df_bus = bus_df(bus)
    m_days = movement_days(bus)
    fig, axs = plt.subplots(math.ceil(len(m_days)/3),3, figsize=(16,4*math.ceil(len(m_days)/3)), squeeze=False)
    for i in range(len(m_days)):
        df_temp = df_bus[df_bus['datetime'].dt.dayofyear == m_days[i]]
        sns.scatterplot(data=df_temp, x='longitude', y='latitude', ax=axs[i//3][i%3], label=m_days[i])
    plt.show()
    

def animate_df(df):
    x_data = df['longitude'].values
    y_data = df['latitude'].values

    plt.plot(x_data, y_data, 'o')
    ax_d = plt.gca()
    plt.close()

    fig, ax = plt.subplots()
    ax.set_xlim(ax_d.get_xlim())
    ax.set_ylim(ax_d.get_ylim())

    lines = plt.plot([],'o')
    line = lines[0]

    def animate(frame):
        x = x_data[:frame]
        y = y_data[:frame]
        line.set_data((x,y))

    anim = FuncAnimation(fig, func=animate, frames=len(x_data), interval=100)
    video = anim.to_html5_video()
    html = display.HTML(video)
    display.display(html)
    plt.close()
    
    
def bus_movement_days():
    buses = df_full['bus_id'].unique()
    m_days = {}
    for bus in buses:
        m_days[bus] = movement_days(bus)
    return m_days

def distance(lat1, lon1, lat2, lon2):
    R = 6371000
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (math.sin(math.pi/180 * dlat/2))**2 + math.cos(math.pi /180 * lat1) * math.cos(math.pi/180 * lat2) * (math.sin(math.pi/180*dlon/2))**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a) )
    d = R * c
    return d

def distances(lons1, lats1, lons2, lats2):
    R = 6371000
    dlon = lons2 - lons1
    dlat = lats2 - lats1
    a = (np.sin(np.pi/180 * dlat/2))**2 + np.cos(np.pi /180 * lats1) * np.cos(np.pi/180 * lats2) * (np.sin(np.pi/180*dlon/2))**2
    c = 2 * np.arctan(np.sqrt(a), np.sqrt(1-a) )
    d = R * c
    return d
    
def minimum_distance(lat, lon, lats2, lons2):
    R = 6371000
    l = np.ones(len(lats2))
    lats1 = l*lat
    lons1 = l*lon
    dlon = lons2 - lons1
    dlat = lats2 - lats1
    a = (np.sin(np.pi/180 * dlat/2))**2 + np.cos(np.pi /180 * lats1) * np.cos(np.pi/180 * lats2) * (np.sin(np.pi/180*dlon/2))**2
    c = 2 * np.arctan(np.sqrt(a), np.sqrt(1-a) )
    d = R * c
    i = np.where(d == min(d))[0][0]
    return [transit_station_names[i], min(d)]

def minimum_distances(lats, lons):
    distances = []
    names = []
    for i in range(len(lats)):
        [name, d] = minimum_distance(lats[i], lons[i], np.array(transit_station_lats), np.array(transit_station_lons))
        distances.append(d)
        names.append(name)
    return [distances, names]

def nearest_station(lon, lat, stations):
    R = 6371000
    lats2 = stations['latitude'].values
    lons2 = stations['longitude'].values
    l = np.ones(len(lats2))
    lats1 = l*lat
    lons1 = l*lon
    dlon = lons2 - lons1
    dlat = lats2 - lats1
    a = (np.sin(np.pi/180 * dlat/2))**2 + np.cos(np.pi /180 * lats1) * np.cos(np.pi/180 * lats2) * (np.sin(np.pi/180*dlon/2))**2
    c = 2 * np.arctan(np.sqrt(a), np.sqrt(1-a) )
    d = R * c
    i = np.where(d == min(d))[0][0]
    nearest_station = stations.iloc[i]
    return [min(d), nearest_station['name'], [nearest_station['longitude'], nearest_station['latitude']]]

def nearest_stations(route, stations, maxDistance=100, drop_duplicates=True):
    distance_to_nearest_station = []
    nearest_station_name = []
    latitude_of_nearest_station = []
    longitude_of_nearest_station = []
    
    lons = route['longitude'].values
    lats = route['latitude'].values
    for i in range(route.shape[0]):
        [d, name, [longitude, latitude]] = nearest_station(lons[i], lats[i], stations)
        distance_to_nearest_station.append(d)
        nearest_station_name.append(name)
        longitude_of_nearest_station.append(longitude)
        latitude_of_nearest_station.append(latitude)
        
    route['name']= nearest_station_name
    route['nlongitude'] = longitude_of_nearest_station
    route['nlatitude'] = latitude_of_nearest_station
    route['distance'] = distance_to_nearest_station
        
    if drop_duplicates:
        route = route[route['distance'] < maxDistance]
        return route.drop_duplicates(subset='name')
    else:
        return route

def parse_kml(filename):
    with open(filename, "r") as content:
        soup = BeautifulSoup(content, "lxml-xml")
    route_name = soup.find('name').text
    full_route = soup.find('description').text
    folders = soup.findAll('Folder')
    stop_names = []
    stop_coordinates = []
    
    route_coordinates = []
    
    for folder in folders:
        folder_name = folder.find('name').text.split(' ')[0]
        if folder_name == 'Bus':
            for point in folder.findAll('Placemark'):
                stop_name = point.find('name').text
                coordinate = point.find('coordinates').text
                stop_names.append(stop_name)
                stop_coordinates.append(coordinate)
        
        elif folder_name == 'Directions':
            route_coordinates = folder.find('LineString').coordinates.text.split('\n')
            
    if len(stop_coordinates)>1:
        stop_coordinates = [coordinate.split(',')[:-1] for coordinate in stop_coordinates]
        stop_lons = np.array([float(coordinate[0]) for coordinate in stop_coordinates])
        stop_lats = np.array([float(coordinate[1]) for coordinate in stop_coordinates])
   
    if len(route_coordinates)>1:
        route_coordinates = [coordinate.strip() for coordinate in route_coordinates][1:-1]
        route_coordinates = [coordinate.split(',')[:-1] for coordinate in route_coordinates]
        route_lons = np.array([float(coordinate[0]) for coordinate in route_coordinates])
        route_lats = np.array([float(coordinate[1]) for coordinate in route_coordinates])
        df = pd.DataFrame({'latitude': route_lats, 'longitude': route_lons})
    
    route_coordinates = pd.DataFrame({'longitude': route_lons, 'latitude': route_lats})
    if len(stop_coordinates)>1:
        route_stops = pd.DataFrame({'names': stop_names, 'longitude': stop_lons, 'latitude': stop_lats})    
        return [route_name, full_route, route_stops, route_coordinates]
    return [route_name, full_route, route_coordinates]

def check_endpoint(df, endpoint0, endpoint1, cropToEndpoints=True):
    df.sort_values(['last_update'], inplace=True)
    df['Endpoint0'] = df['name']==endpoint0
    df['Endpoint1'] = df['name']==endpoint1
    df['Endpoint'] = np.logical_or(df['Endpoint0'], df['Endpoint1'])
    df.drop(columns=['Endpoint0', 'Endpoint1'], inplace=True)
    if cropToEndpoints:
        start = df[df['Endpoint']==True].index[0]
        end = df[df['Endpoint']==True].index[-1]
        df = df.loc[start:end]
    df.reset_index(inplace=True)
    df.drop(columns=['index'], inplace=True)
    
    return df

def generate_tour_endpoints(df):
    tour_endpoints = np.where(np.logical_xor(df['Endpoint'][:-1].values, df['Endpoint'][1:].values) == True)[0]
    
    tour_endpoints[1::2]+=1
    names = df.iloc[tour_endpoints]['name'].values
    indices = []
    for i in range(len(names)):
        if i==0:
            if names[i]!=names[i+1]:
                indices.append(i)
        elif i==len(names)-1:
            if names[i]!=names[i-1]:
                indices.append(i)
        else:
            if (names[i] != names[i-1] or names[i]!=names[i+1]):
                indices.append(i)
    tour_endpoints = tour_endpoints[indices]

    return tour_endpoints

def generate_tours(df, endpoint0, endpoint1):
    df_tmp = check_endpoint(df, endpoint0, endpoint1)
    tour_endpoints = generate_tour_endpoints(df_tmp)
    dfs = []

    for i in range(0,len(tour_endpoints), 2):
        df_temp = df_tmp.iloc[tour_endpoints[i]: tour_endpoints[i+1]+1]
        if df_temp.shape[0]>20:
            dfs.append(df_temp)
    return dfs

def generate_training_data(df_tmp):
    df_tmp_bus_stops = df_tmp.iloc[1:-1]['distance']<100
    df_tmp_bus_stops = df_tmp_bus_stops.values
    df_tmp_bus_stops = np.concatenate(([True], df_tmp_bus_stops, [True]))
    df_tmp = df_tmp[df_tmp_bus_stops]
    names = df_tmp['name'].values
    indices = [0]
    for i in range(1,len(names)-1):
        if (names[i] != names[i-1] or names[i]!=names[i+1]):
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

    df = pd.DataFrame({'From':names_from, 'To': names_to, 'Start': time_from, 'End': time_to, 'Distance': d})
    df['Time_Seconds'] = (df['End'] - df['Start']).dt.seconds

    return df

def generate_training_data_alt(df_tmp):
    df_tmp_bus_stops = df_tmp.iloc[1:-1]['distance']<100
    df_tmp_bus_stops = df_tmp_bus_stops.values
    df_tmp_bus_stops = np.concatenate(([True], df_tmp_bus_stops, [True]))
    df_tmp = df_tmp[df_tmp_bus_stops]
    names = df_tmp['name'].values
    indices = [0]
    for i in range(1,len(names)-1):
        if (names[i] != names[i-1] or names[i]!=names[i+1]):
            indices.append(i)
    indices.append(i+1)
    df_tmp = df_tmp.iloc[indices]
    names = df_tmp['name'].values
    times = df_tmp['last_update'].values
    
    dfs = []
    
    for i in range(len(names)-1):
        from_ = pd.DataFrame({'from': names[i:i+1], 'Start': times[i:i+1]})
        l = len(names)-1-i
        end = len(names)
        
        df = pd.concat([from_]*l)
        df['to'] = names[i+1:end]
        df['End'] = times[i+1:end]
        df.drop_duplicates(subset=['from','to'], inplace=True)
        if df.iloc[0]['from'] == df.iloc[0]['to']:
            continue
        dfs.append(df)
    return pd.concat(dfs)

def route_final(route, df):
    if route=='lagankhel-nayabuspark' or route=='nayabuspark-lagankhel' or route=='Lagankhel-NayaBusPark' or route=='Lagankhel-Buspark':
        stations_in_route = parse_kml('./Bus Routes/Lagankhel-NayaBusPark.kml', getStops=True, drop_duplicates=True, maxDistance=50)
        df = nearest_stations(df, stations_in_route, drop_duplicates=False)
        df.drop_duplicates(subset='last_update', inplace=True)
        stations_in_route.reset_index(inplace=True)
        stations_in_route = stations_in_route.append({'name': 'Lainchaur Bus Stop', 'latitude': 27.716467, 'longitude': 85.315898, 'distance':0 }, ignore_index=True)
        stations_in_route = stations_in_route.append({'name': 'Panipokhari Bus Stop', 'latitude': 27.728172, 'longitude': 85.324627,'distance':0 }, ignore_index=True)
        stations_in_route= stations_in_route.iloc[[0, 1, 4, 5, 6, 8, 9, 10, 12, -2, 13, -1, 15, 17,18,21,23,24,25]]
        a = [bus for bus in buses.loc[87]['route_full'].split('-')]
        a.insert(-3,'Chauki (Basundhara)')
        stations_in_route['name'] = a
        return stations_in_route
    
def generate_training_data_for_bus(bus):
    dfs_outer = []
    df = bus_df(bus)
    days = movement_days(bus)
    for day in days:
        df = bus_day_df(bus, day)
        df.drop_duplicates(subset='last_update', inplace=True)
        stations_in_route = route_final('lagankhel-nayabuspark', df)
        df = nearest_stations(df, stations_in_route, drop_duplicates=False)
        dfs = []
        tours = generate_tours(df, stations_in_route['name'].iloc[0], stations_in_route['name'].iloc[-1])
        print('Bus: {} - Day: {} - Tours: {}  '.format(bus,day, len(tours)))
        for tour in tours:
            df = generate_training_data_alt(tour)
            dfs.append(df)
        if len(dfs)<1:
            continue
        df_outer = pd.concat(dfs)
        dfs_outer.append(df_outer)
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

def generate_training_data_for_buses(buses):
    dfs_outer = []
    for bus in buses:
        df = generate_training_data_for_bus(bus)
        dfs_outer.append(df)
    train = pd.concat(dfs_outer)
    segments = {}
    for segment in train.index.unique():
        segments[segment] = train.loc[segment]
    return segments