import requests
import json
from datetime import datetime, timezone
import pytz
import re
from urllib import parse
from apscheduler.schedulers.background import BackgroundScheduler
import os
from bs4 import BeautifulSoup
import numpy as np

import pandas as pd
from flask import Flask, jsonify, request

import math

models = pd.read_pickle('models.pkl')
stops = list(set([stop.split('-')[0].strip() for stop in models.index.values]))

app = Flask(__name__)

buses = pd.read_json('buses.json')

UPDATE_NUMBER = 0

routes = {}
dfs = {}
length = 13 * 60 * 60 // 25
datetimes = [datetime.now()] * length
longitudes = [85.0] * length
latitudes = [27.0] * length
last_update_times = [datetime.now()] * length
fix_times = [datetime.now()] * length
nearest_stops = ['unknown'] * length
active = [False] * length
trips = ['unknown'] * length
for bus in buses.index.values:
    dfs[str(bus)] = pd.DataFrame({'datetime': datetimes, 'longitude': longitudes, 'latitude': latitudes,
                                  'last_update': last_update_times, 'fix_time': fix_times, 'nearest_stop': nearest_stops, 'active': active, 'trip': trips})


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


routes['Godawari-Ratnapark'] = parse_kml('./Bus Routes/Godawari-Ratnapark.kml')
routes['Lagankhel-Budhanilakantha'] = parse_kml(
    './Bus Routes/Lagankhel-Budhanilakantha.kml')
routes['Lagankhel-NayaBuspark'] = parse_kml(
    './Bus Routes/Lagankhel-NayaBuspark.kml')
routes['Lamatar-Ratnapark'] = parse_kml('./Bus Routes/Lamatar-Ratnapark.kml')
routes['Thankot-Airport'] = parse_kml('./Bus Routes/Thankot-Airport.kml')


def getData():
    r = requests.post(
        "http://117.121.237.226:83/ambulance/api/temp_api",
        headers={"Authorization": "Bearer abcdefghij"},
    )
    data = r.json()
    devices = data["data"][0]["devices"]
    return devices


def getBuses(devices):
    buses_list = []
    for device in devices:
        name_route = " ".join(device["name"].split())
        name = name_route[:17]
        route = name_route[17:]
        route = re.sub("[() ]", "", route.strip())
        full_route = re.sub("[()]", "", device["model"])
        bus = {
            "bus_id": device["id"],
            "bus_name": name,
            "route_short": route,
            "route_full": full_route,
            "phone": device["phone"],
        }
        buses_list.append(bus)
    return buses_list


def getPositions(devices):
    positions_list = []
    tz = pytz.timezone("Asia/Kathmandu")
    kathmandu_now = datetime.now(tz)
    for device in devices:
        if device["position"]:
            position = {
                "bus_id": device["position"]["deviceid"],
                "datetime": kathmandu_now,
                "latitude": device["position"]["latitude"],
                "longitude": device["position"]["longitude"],
                "last_update": datetime.strptime(
                    device["lastupdate"], "%Y-%m-%d %H:%M:%S"
                ),
                "fixtime": datetime.strptime(
                    device["position"]["fixtime"], "%Y-%m-%d %H:%M:%S"
                ),
            }
        positions_list.append(position)
    return positions_list


def calculate_distance(lat1, lon1, lat2, lon2):
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
    return nearest_station['name']


def add_buses_and_positions():
    data = getData()
    buses_data = getBuses(data)
    positions_data = getPositions(data)
    engine.execute(positions.insert(), positions_data)
    engine.execute(buses.insert(), buses_data)


def add_positions():
    data = getData()
    positions_data = getPositions(data)
    engine.execute(positions.insert(), positions_data)
    print("Position added.")


def add_positions_five():
    global UPDATE_NUMBER
    global dfs
    data = getData()
    positions_data = getPositions(data)
    for bus in positions_data:
        bus_id = bus['bus_id']
        route_name = buses.loc[bus_id]['route_short']
        bus_id = str(bus_id)

        stations = routes[route_name]['stops']
        endpoint0 = stations['name'].iloc[0]
        endpoint1 = stations['name'].iloc[-1]

        route_name_alt = route_name.split('-')
        route_name_alt = [route_name_alt[-1], route_name_alt[0]]
        route_name_alt = "-".join(route_name_alt)

        df = dfs[str(bus['bus_id'])]

        nearest_station = generate_nearest_station(
            bus['longitude'], bus['latitude'], stations)

        dfs[bus_id].iloc[UPDATE_NUMBER, 0] = bus['datetime']
        dfs[bus_id].iloc[UPDATE_NUMBER, 1] = bus['longitude']
        dfs[bus_id].iloc[UPDATE_NUMBER, 2] = bus['latitude']
        dfs[bus_id].iloc[UPDATE_NUMBER, 3] = bus['last_update']
        dfs[bus_id].iloc[UPDATE_NUMBER, 4] = bus['fixtime']
        dfs[bus_id].iloc[UPDATE_NUMBER, 5] = nearest_station

        if UPDATE_NUMBER > 3:
            distance = calculate_distance(
                bus['latitude'], bus['longitude'], df['latitude'].iloc[UPDATE_NUMBER-2], df['longitude'].iloc[UPDATE_NUMBER-2])
            if distance > 0:
                dfs[bus_id].iloc[UPDATE_NUMBER, 6] = True

            if df.iloc[UPDATE_NUMBER-1, 5] != endpoint0 and df.iloc[UPDATE_NUMBER-1, 5] != endpoint1:  # if not endpoint
                if nearest_station == endpoint0 or nearest_station == endpoint1:  # if endpoint
                    dfs[bus_id].iloc[UPDATE_NUMBER, 7] = 'unknown'
                else:
                    dfs[bus_id].iloc[UPDATE_NUMBER,
                                     7] = df.iloc[UPDATE_NUMBER - 1, 7]
            else:
                if df.iloc[UPDATE_NUMBER, 5] == endpoint0 or df.iloc[UPDATE_NUMBER, 5] == endpoint1:
                    dfs[bus_id].iloc[UPDATE_NUMBER, 7] == 'unknown'
                else:
                    if df.iloc[UPDATE_NUMBER-1, 5] == endpoint0:
                        dfs[bus_id].iloc[UPDATE_NUMBER, 7] == route_name
                    else:
                        dfs[bus_id].iloc[UPDATE_NUMBER, 7] = route_name_alt

    print("Position added {}".format(UPDATE_NUMBER))
    UPDATE_NUMBER += 1


def calculate_time(A, B, dt):
    if A not in stops:
        return 'Data not available for {}'.format(A)
    if B not in stops:
        return 'Data not available for {}'.format(B)
    segment = A + ' - ' + B
    hour = dt.strftime('%H')
    month = dt.strftime('%m')
    weekday = dt.strftime('%w')
    day = dt.strftime('%d')
    X = [[month, day, hour, weekday]]
    model = models.loc[segment]['RandomForestModel']
    t = model.predict(X)[0]
    return 'It takes {} seconds to reach from {} to {}.'.format(t, A, B)


@app.route('/')
def home():
    return 'Home'


@app.route('/stops')
def bus_stops():
    return jsonify(stops)


@app.route('/check', methods=['POST'])
def check():
    bus_id = str(request.args.get('bus_id'))

    # return {'datetime': dfs[bus_id].iloc[UPDATE_NUMBER, 0],
    #         'latitude': dfs[bus_id].iloc[UPDATE_NUMBER, 1],
    #         'longitude': dfs[bus_id].iloc[UPDATE_NUMBER, 2],
    #         'last_update': dfs[bus_id].iloc[UPDATE_NUMBER, 3],
    #         'fixtime': dfs[bus_id].iloc[UPDATE_NUMBER, 4],
    #         ''}

    return dfs[bus_id].iloc[UPDATE_NUMBER-1].to_json()


@ app.route('/predict', methods=['POST'])
def predict():
    A = request.args.get('from')
    B = request.args.get('to')
    present = datetime.now()
    return calculate_time(A, B, present)


if __name__ == "__main__":

    scheduler = BackgroundScheduler()

    scheduler.add_job(
        add_positions_five,
        trigger="cron",
        second="*/5",
        minute="*",
        hour="*",
        day="*",
        week="*",
        year="*",
        day_of_week="*",
    )

    scheduler.start()

    app.run()
