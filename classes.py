import pandas as pd


class Vessel:
    def __init__(self, name):
        table = pd.read_excel('C:/dev/data/speed_table.xlsx', sheet_name=name)
        self.name = name
        self.speeds = table['Speed']
        self.fuel_rates = {speed: table['Fuel'][idx] for idx, speed in enumerate(table['Speed'])}
