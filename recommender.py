# Maintained by jemo from 2021.4.12 to now
# Created by jemo on 2021.4.12 17:35:49
# Recommender

from sqlalchemy import create_engine
import pymysql
import pandas as pd
from config import db_connection

conn = create_engine(db_connection)

df = pd.read_sql("select * from imageBrowseRecord", conn)

print("df: ", df)
