import sqlite3
import pandas as pd
import os
import json

def erd_text_generate(name, replacement=None):
    """
    Generate schema information and ERD text from a SQLite database.

    Args:
        name (str): database name (without .sqlite extension).
        replacement (dict): {table: {original_col: [expanded_cols...]}}
                            If provided, schema_info will replace those columns,
                            and ERD text will be generated from updated schema_info.
    """
    database_path = name + ".sqlite"
    conn = sqlite3.connect(database_path)

    # Step 1: List tables
    tables_df = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)
    tables = tables_df['name'].tolist()

    schema_info = {}

    for table in tables:
        # Get columns and types
        columns = pd.read_sql(f"PRAGMA table_info({table});", conn)
        # Get foreign keys
        foreign_keys = pd.read_sql(f"PRAGMA foreign_key_list({table});", conn)

        # Prepare column list
        col_list = []
        for _, row in columns.iterrows():
            col_name, col_type = row["name"], row["type"]

            if replacement and table in replacement and col_name in replacement[table]:
                # Replace with expanded columns
                for exp_col in replacement[table][col_name]:
                    col_list.append({"name": exp_col, "type": "INT"})
            else:
                col_list.append({"name": col_name, "type": col_type})

        schema_info[table] = {
            "columns": col_list,
            "primary_keys": columns[columns['pk'] > 0]['name'].tolist(),
            "foreign_keys": foreign_keys[['from', 'table', 'to']].to_dict(orient="records")
        }

    conn.close()

    # ---- Generate ERD text from updated schema_info ----
    def generate_erd_text(schema_info):
        erd_text = ""
        for table, info in schema_info.items():
            erd_text += f"\nTable: {table}\n"
            erd_text += "Columns:\n"
            for col in info["columns"]:
                erd_text += f"  - {col['name']} ({col['type']})\n"

            if info["primary_keys"]:
                erd_text += f"Primary Key: {', '.join(info['primary_keys'])}\n"
            if info["foreign_keys"]:
                erd_text += "Foreign Keys:\n"
                for fk in info["foreign_keys"]:
                    erd_text += f"  - {fk['from']} → {fk['table']}.{fk['to']}\n"
        return erd_text

    erd_text = generate_erd_text(schema_info)

    return schema_info, erd_text