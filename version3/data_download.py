import requests
import psycopg2
import polars as pl
import datetime
import socket
import subprocess


now = datetime.datetime.now()
year = now.strftime("%Y")
month = now.strftime("%m")
day = now.strftime("%d")

start_date = "2022-01-01"
end_date = f"{year}-{month}-{day}"

dbname = 'ai_dataset'
dbuser = 'postgres'
dbpassword = 'parola'
dbport = 5432
dbhost = '127.0.0.1'
conn = psycopg2.connect(dbname=dbname, user=dbuser, password=dbpassword, host=dbhost, port=dbport)  

def unix_to_time(timestamp):
    dt_object=datetime.datetime.fromtimestamp(timestamp)
    while dt_object.weekday()>5:
        dt_object-=datetime.timedelta(days=1)
    time=dt_object.strftime('%Y,%m,%d')
    
    return time

def replace_keys_in_json_files(data, name):
    """Replaces keys in multiple JSON files based on a key mapping.

    Args:
        filenames: A list of file paths to the JSON files.
        key_map: A dictionary mapping old keys to their new replacements.
    """
    char_to_replacement_map = {
        'o': f'{name}_Open',
        'h': f'{name}_High',
        'l': f'{name}_Low',
        'c': f'{name}_Close',
        'a': f'{name}_Adj_Close',
        'v': f'{name}_Volume'
    }

    # Iterate through the list and replace keys within dictionaries
    for i, item in enumerate(data):
        if isinstance(item, dict):  # Check if the item is a dictionary
            data[i] = {char_to_replacement_map.get(key, key): value for key, value in item.items()}
        else: 
            print('Error in key replacement')
            # Handle non-dictionary items (you can choose to skip them, raise an error, etc.)
    return data      

def tickers2(tickers):
    replaced = []
    for i in range(len(tickers)): 
        replaced.append(tickers[i].replace('.', '_'))
#        if tickers[i][0].isdigit():
#            'a'+tickers[i][0]
    return replaced

approve = open("log.txt", "r")
if not approve.read() == end_date:

    print("Attempting to download data")


    api_key = 'dd75440b5984491b9f3593d8cc275ed4'
    #api_key='348c590f9b0248638d37b8f381287cf1'
    data_json = []
    tickers = ["AAPL", "VIX.INDX", 'SPY', 'XAUUSD.OANDA', 'BCO.ICMTRADER','NVDA']

    try:
        with socket.create_connection(("www.youtube.com",80),timeout=5) as connection:
            pass
        try:
            with socket.create_connection(("api.darqube.com",80),timeout=5) as connection:
                pass
        except:
            print("Failed to connect to api domain , Internet connection is OK")
    except:
        print("Failed to connect to the internet , check internet connection")
        exit()

    x = tickers2(tickers)
    concatanated_data = []
    for i in range(len(tickers)):
        try:
            response = requests.get(f'https://api.darqube.com/data-api/market-data/historical/daily/{tickers[i]}?token={api_key}&start_date={start_date}&end_date={end_date}')
            response.raise_for_status()
            data_json = response.json()
            data_json = replace_keys_in_json_files(data_json, x[i])
            concatanated_data.append(data_json)
        except requests.exceptions.RequestException as e:
            print(f"API Error: {e}")
    
    table_name = 'tabela'
    cursor = conn.cursor()


    reset_table=f"TRUNCATE TABLE {table_name}"
    cursor.execute(reset_table)
    
    for lists in concatanated_data:
        first_iteration = True
        for sub_list in lists:
            if first_iteration:
                keys = sub_list.keys()
                keys = [item.replace(' ', '_') for item in keys]
#                for i in range(len(keys)):
#                   if  keys[i][0].isdigit():
#                        'a'+keys[i]

                # Create table dynamically based on the keys
                column_definitions = ", ".join([f"ADD COLUMN IF NOT EXISTS {key} REAL " for key in keys])
                create_table_query = f"""
                    ALTER TABLE {table_name}
                    {column_definitions}
                    ;
                """
                cursor.execute(create_table_query)
            first_iteration = False

            value = list(sub_list.values())

            # Prepare the INSERT query dynamically
            insert_query = f"""
                INSERT INTO {table_name} ({", ".join(keys)})
                VALUES({[number for number in value]})
            """
            insert_query = insert_query.replace('[', '')
            insert_query = insert_query.replace(']', '')
            cursor.execute(insert_query)

    conn.commit()
    
    data=pl.read_database(query="SELECT t FROM tabela ORDER BY t ASC",connection=conn)
    time=[]
    data=data['t']
    
    for timp in data:
        time.append(unix_to_time(timp))
    
    cursor.execute("ALTER TABLE tabela ADD COLUMN IF NOT EXISTS timp DATE")
    
    for i in range(len(time)):
        sql_update_query="UPDATE tabela SET timp= %s WHERE t = %s;"
        cursor.execute(sql_update_query,(time[i], data[i]))
        

    conn.commit()
    conn.close()
    print("data downloaded")
    print(end_date)
    approve = open("log.txt", "w")
    approve.write(end_date)
    approve.close()
    

else:
    print("did not download")
