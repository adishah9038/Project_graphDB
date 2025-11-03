import sqlite3
import pandas as pd
import os
from datetime import datetime

def export_sqlite_to_csv(db_path: str, output_dir: str):
    """
    Export all tables from a SQLite database into CSV files.
    
    Rules:
    - VARCHAR, CHAR -> str
    - INT, SMALLINT, NUMERIC -> int64
    - DECIMAL -> float64
    - BLOB -> object
    - TIMESTAMP/DATE -> split into year, month, day, hour, minute, second (int64)

    Returns:
    dict -> {table_name: {original_col: [expanded_cols...]}}
    """
    os.makedirs(output_dir, exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # fetch all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]

    replaced_cols_info = {}

    for table in tables:
        print(f"Processing table: {table}")

        # get schema info
        cursor.execute(f"PRAGMA table_info({table});")
        schema = cursor.fetchall()  # (cid, name, type, notnull, dflt_value, pk)

        # read table
        df = pd.read_sql_query(f"SELECT * FROM {table};", conn)

        # process columns
        new_cols = {}
        table_replacements = {}

        for col_id, col_name, col_type, *_ in schema:
            col_type = col_type.upper()
            if col_name not in df.columns:
                continue

            if "CHAR" in col_type or "VARCHAR" in col_type or "TEXT" in col_type:
                df[col_name] = df[col_name].astype(str)

            elif "INT" in col_type or "NUMERIC" in col_type:
                df[col_name] = pd.to_numeric(df[col_name], errors="coerce").astype("Int64")

            elif "DECIMAL" in col_type or "REAL" in col_type or "FLOAT" in col_type:
                df[col_name] = pd.to_numeric(df[col_name], errors="coerce").astype("float64")

            elif "BLOB" in col_type:
                df[col_name] = df[col_name].astype(object)

            elif "DATE" in col_type or "TIME" in col_type:
                # expand timestamp into components
                dt_series = pd.to_datetime(df[col_name], errors="coerce")
                expanded_cols = [
                    f"{col_name}_year",
                    f"{col_name}_month",
                    f"{col_name}_day",
                    f"{col_name}_hour",
                    f"{col_name}_minute",
                    f"{col_name}_second"
                ]
                new_cols[expanded_cols[0]] = dt_series.dt.year.astype("Int64")
                new_cols[expanded_cols[1]] = dt_series.dt.month.astype("Int64")
                new_cols[expanded_cols[2]] = dt_series.dt.day.astype("Int64")
                new_cols[expanded_cols[3]] = dt_series.dt.hour.astype("Int64")
                new_cols[expanded_cols[4]] = dt_series.dt.minute.astype("Int64")
                new_cols[expanded_cols[5]] = dt_series.dt.second.astype("Int64")
                df.drop(columns=[col_name], inplace=True)

                # record replacements
                table_replacements[col_name] = expanded_cols

        # add expanded timestamp columns
        for new_col, series in new_cols.items():
            df[new_col] = series

        # save to CSV
        csv_path = os.path.join(output_dir, f"{table}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved {table} -> {csv_path}")

        if table_replacements:
            replaced_cols_info[table] = table_replacements

    conn.close()
    print("All tables exported!")

    return replaced_cols_info