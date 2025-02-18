"""
This module provides a MySQLConnector class for executing SQL queries
and returning results in JSON format.
"""

import json
import datetime
import mysql.connector


class MySQLConnector:
    """A class to connect to a MySQL database and execute SQL queries."""

    def __init__(self, host: str, user: str, password: str, database: str):
        self.connection = mysql.connector.connect(host=host, user=user, password=password, database=database)

    def execute_sql_query(self, sql: str) -> str:
        """Executes a SQL query and returns the result in JSON format."""
        sql = sql.replace("\\n", " ")
        connection = None
        cursor = None
        try:
            # 连接到MySQL数据库
            cursor = self.connection.cursor()
            cursor.execute(sql)
            columns = [column[0] for column in cursor.description]
            result = cursor.fetchall()
            # Convert date objects to strings
            result_dict = [
                {
                    column: (value.isoformat() if isinstance(value, (datetime.date, datetime.datetime)) else value)
                    for column, value in zip(columns, row)
                }
                for row in result
            ]
            return json.dumps(result_dict, ensure_ascii=False)
        except mysql.connector.Error as err:
            return f"Error: {err}"
        finally:
            if connection is not None and connection.is_connected():
                cursor.close()
                connection.close()
