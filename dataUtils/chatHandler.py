"""
    Contains Data Utility functions to collect incoming datasets
"""
import sqlite3
import pandas as pd
from typing import Dict, List 

def storeBase(new_data: List[Dict[str, float]], table_name: str, save_path: str):  
    df = pd.DataFrame(new_data)
    connection = sqlite3.connect(save_path)
    
    df.to_sql(table_name, connection, if_exists='append', index=False)
    connection.commit()
    connection.close()

def loadBase(load_path: str, connect: sqlite3 = None):
    connection = connect if connect is not None else sqlite3.connect(load_path)
    df_from_db = pd.read_sql('SELECT * FROM emotions', connection)
    print(df_from_db.head(5))
    return df_from_db

def complete_session(base: object, save_folder: str):
    """Saves data after completing chat session."""
    import os 
    from uuid import uuid4
    from datetime import datetime 
    import numpy as np 
    current_date = datetime.now().strptime("%m%Y")
    session_id = uuid4().hex
    session_result = base.session 
    # session_metric = base.breed_belief 
    np.savez(f"{save_folder}/{current_date}/{session_id}.npz", *session_result)
