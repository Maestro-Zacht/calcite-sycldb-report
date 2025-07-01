import duckdb as ddb
import numpy as np
import pandas as pd
import time
from collections import defaultdict


'''
file = '/data/ssd/mnt/ivan/db_ops/ock/target8/build_client/run_with_acpp/sycldb-1.0/ssb/data/s20_columnar/CUSTOMER3'
data = np.fromfile(file, dtype=np.int32)
df = pd.DataFrame(data)
print(df.head())
ddb.sql("CREATE TABLE my_table AS SELECT * FROM df")
ddb.sql("INSERT INTO my_table SELECT * FROM df")
'''


def index_of(arr, val):
    """Finds the index of a value in a list. Returns -1 if not found."""
    try:
        return arr.index(val)
    except ValueError:
        return -1


def lookup(col_name):
    """Looks up the column name and returns a formatted string based on the column category."""

    lineorder = ["lo_orderkey", "lo_linenumber", "lo_custkey", "lo_partkey",
                 "lo_suppkey", "lo_orderdate", "lo_orderpriority",
                 "lo_shippriority", "lo_quantity", "lo_extendedprice",
                 "lo_ordtotalprice", "lo_discount", "lo_revenue", "lo_supplycost",
                 "lo_tax", "lo_commitdate", "lo_shipmode"]

    part = ["p_partkey", "p_name", "p_mfgr", "p_category", "p_brand1",
            "p_color", "p_type", "p_size", "p_container"]

    supplier = ["s_suppkey", "s_name", "s_address", "s_city",
                "s_nation", "s_region", "s_phone"]

    customer = ["c_custkey", "c_name", "c_address", "c_city",
                "c_nation", "c_region", "c_phone", "c_mktsegment"]

    date = ["d_datekey", "d_date", "d_dayofweek", "d_month", "d_year",
            "d_yearmonthnum", "d_yearmonth", "d_daynuminweek",
            "d_daynuminmonth", "d_daynuminyear", "d_sellingseason",
            "d_lastdayinweekfl", "d_lastdayinmonthfl", "d_holidayfl", "d_weekdayfl"]

    if col_name[0] == 'l':
        index = index_of(lineorder, col_name)
        return f"LINEORDER{index}" if index != -1 else ""

    elif col_name[0] == 's':
        index = index_of(supplier, col_name)
        return f"SUPPLIER{index}" if index != -1 else ""

    elif col_name[0] == 'c':
        index = index_of(customer, col_name)
        return f"CUSTOMER{index}" if index != -1 else ""

    elif col_name[0] == 'p':
        index = index_of(part, col_name)
        return f"PART{index}" if index != -1 else ""

    elif col_name[0] == 'd':
        index = index_of(date, col_name)
        return f"DDATE{index}" if index != -1 else ""

    return ""

# Example usage
# col_name = "d_daynuminyear"
# result = lookup(col_name)
# print(result)


base_path = '/ssb/s20_columnar/'


def Q11():
    tables = {'lineorder': ['lo_orderdate', 'lo_discount', 'lo_quantity', 'lo_extendedprice']}
    df = pd.DataFrame(columns=tables['lineorder'])
    for table, columns in tables.items():
        for col in columns:
            file_name = lookup(col)
            file_path = base_path + file_name
            data = np.fromfile(file_path, dtype=np.int32)
            df[col] = data

    print(df.head())
    ddb.sql("CREATE TABLE lineorder (lo_orderdate INT, lo_discount INT, lo_quantity INT, lo_extendedprice INT)")
    ddb.sql("INSERT INTO lineorder SELECT * FROM df")
    start_time = time.time()
    result = ddb.sql("select sum(lo_extendedprice * lo_discount) from lineorder \
                      where lo_orderdate >= 19930101 and lo_orderdate <= 19940101 and lo_discount>=1 \
                      and lo_discount<=3 and lo_quantity<25;")
    print(result)
    end_time = time.time()
    print(f"Time taken: {(end_time - start_time) * 1000}")


def Q21():
    tables = {'lineorder': ['lo_orderdate', 'lo_partkey', 'lo_suppkey', 'lo_revenue'],
              'part': ['p_partkey', 'p_brand1', 'p_category'],
              'supplier': ['s_region', 's_suppkey'],
              'ddate': ['d_year', 'd_datekey']}
    dfs = {}
    for table, columns in tables.items():
        df = pd.DataFrame(columns=columns)
        for col in columns:
            file_name = lookup(col)
            file_path = base_path + file_name
            data = np.fromfile(file_path, dtype=np.int32)
            df[col] = data
        cols = ' INT, '.join(columns) + ' INT'
        # check if the table already exists
        ddb.sql(f"CREATE TABLE {table} ({cols})")
        ddb.sql(f"INSERT INTO {table} SELECT * FROM df")
    start_time = time.time()
    result = ddb.sql("select sum(lo_revenue),d_year,p_brand1\
                        from lineorder,part,supplier,ddate\
                        where lo_orderdate = d_datekey\
                        and lo_partkey = p_partkey\
                        and lo_suppkey = s_suppkey\
                        and p_category = 1\
                        and s_region = 1\
                        group by d_year,p_brand1;")
    print(result)
    end_time = time.time()
    print(f"Time taken: {(end_time - start_time) * 1000}")
    # drop the tables
    for table in tables.keys():
        ddb.sql(f"DROP TABLE {table}")


'''
select sum(lo_revenue),d_year,p_brand1
from lineorder,part,supplier,ddate
where lo_orderdate = d_datekey
and lo_partkey = p_partkey
and lo_suppkey = s_suppkey
and p_category = 1
and s_region = 1
group by d_year,p_brand1;
'''

# Q21()
# Q21()

query_tables_columns = {
    11: {'lineorder': ['lo_orderdate', 'lo_discount', 'lo_quantity', 'lo_extendedprice']},
    12: {'lineorder': ['lo_orderdate', 'lo_discount', 'lo_quantity', 'lo_extendedprice']},
    13: {'lineorder': ['lo_orderdate', 'lo_discount', 'lo_quantity', 'lo_extendedprice']},
    21: {'lineorder': ['lo_orderdate', 'lo_partkey', 'lo_suppkey', 'lo_revenue'],
         'part': ['p_partkey', 'p_brand1', 'p_category'],
         'supplier': ['s_region', 's_suppkey'],
         'ddate': ['d_year', 'd_datekey']},
    22: {'lineorder': ['lo_orderdate', 'lo_partkey', 'lo_suppkey', 'lo_revenue'],
         'part': ['p_partkey', 'p_brand1'],
         'supplier': ['s_region', 's_suppkey'],
         'ddate': ['d_year', 'd_datekey']},
    23: {'lineorder': ['lo_orderdate', 'lo_partkey', 'lo_suppkey', 'lo_revenue'],
         'part': ['p_partkey', 'p_brand1'],
         'supplier': ['s_region', 's_suppkey'],
         'ddate': ['d_year', 'd_datekey']},
    31: {'lineorder': ['lo_orderdate', 'lo_partkey', 'lo_suppkey', 'lo_revenue'],
         'customer': ['c_region', 'c_custkey', 'c_nation'],
         'supplier': ['s_region', 's_suppkey', 's_nation'],
         'ddate': ['d_year', 'd_datekey']},
    32: {'lineorder': ['lo_orderdate', 'lo_partkey', 'lo_suppkey', 'lo_revenue'],
         'customer': ['c_city', 'c_custkey', 'c_nation'],
         'supplier': ['s_city', 's_suppkey', 's_nation'],
         'ddate': ['d_year', 'd_datekey']},
    33: {'lineorder': ['lo_orderdate', 'lo_partkey', 'lo_suppkey', 'lo_revenue'],
         'customer': ['c_city', 'c_custkey'],
         'supplier': ['s_city', 's_suppkey'],
         'ddate': ['d_year', 'd_datekey']},
    34: {'lineorder': ['lo_orderdate', 'lo_partkey', 'lo_suppkey', 'lo_revenue'],
         'customer': ['c_city', 'c_custkey'],
         'supplier': ['s_city', 's_suppkey'],
         'ddate': ['d_year', 'd_datekey', 'd_yearmonthnum']},
    41: {'lineorder': ['lo_orderdate', 'lo_partkey', 'lo_suppkey', 'lo_revenue', 'lo_custkey', 'lo_supplycost'],
         'customer': ['c_region', 'c_custkey', 'c_nation'],
         'supplier': ['s_region', 's_suppkey'],
         'part': ['p_mfgr', 'p_partkey'],
         'ddate': ['d_year', 'd_datekey']},
    42: {'lineorder': ['lo_orderdate', 'lo_partkey', 'lo_suppkey', 'lo_revenue', 'lo_custkey', 'lo_supplycost'],
         'customer': ['c_region', 'c_custkey'],
         'supplier': ['s_region', 's_suppkey', 's_nation'],
         'part': ['p_mfgr', 'p_partkey', 'p_category'],
         'ddate': ['d_year', 'd_datekey']},
    43: {'lineorder': ['lo_orderdate', 'lo_partkey', 'lo_suppkey', 'lo_revenue', 'lo_custkey', 'lo_supplycost'],
         'customer': ['c_region', 'c_custkey'],
         'supplier': ['s_city', 's_suppkey', 's_nation'],
         'part': ['p_brand1', 'p_partkey'],
         'ddate': ['d_year', 'd_datekey']}
}

table_columns = defaultdict(set)
for query, tables in query_tables_columns.items():
    for table, columns in tables.items():
        table_columns[table].update(columns)

# Uncomment the following lines to create the database
with ddb.connect('ssb.db') as conn:
    for table, columns in table_columns.items():
        df = pd.DataFrame(columns=list(columns))
        for col in columns:
            file_name = lookup(col)
            file_path = base_path + file_name
            data = np.fromfile(file_path, dtype=np.int32)
            df[col] = data
        cols = ' INT, '.join(columns) + ' INT'
        conn.sql(f"CREATE TABLE {table} ({cols})")
        conn.sql(f"INSERT INTO {table} SELECT * FROM df")

queries = {
    11: "select sum(lo_extendedprice * lo_discount) as revenue from lineorder \
        where lo_orderdate >= 19930101 and lo_orderdate <= 19940101 and lo_discount>=1 \
        and lo_discount<=3 and lo_quantity<25;",
    12: "select sum(lo_extendedprice * lo_discount) as revenue \
        from lineorder \
        where lo_orderdate >= 19940101 and lo_orderdate <= 19940131 \
        and lo_discount>=4 and lo_discount<=6 \
        and lo_quantity>=26 \
        and lo_quantity<=35;",
    13: "select sum(lo_extendedprice * lo_discount) as revenue \
        from lineorder \
        where lo_orderdate >= 19940204 \
        and lo_orderdate <= 19940210 \
        and lo_discount>=5 \
        and lo_discount<=7 \
        and lo_quantity>=26 \
        and lo_quantity<=35;",
    21: "select sum(lo_revenue),d_year,p_brand1\
        from lineorder,part,supplier,ddate\
        where lo_orderdate = d_datekey\
        and lo_partkey = p_partkey\
        and lo_suppkey = s_suppkey\
        and p_category = 1\
        and s_region = 1\
        group by d_year,p_brand1\
        order by d_year, p_brand1;",
    22: "select sum(lo_revenue),d_year,p_brand1\
        from lineorder, part, supplier,ddate\
        where lo_orderdate = d_datekey\
        and lo_partkey = p_partkey\
        and lo_suppkey = s_suppkey\
        and p_brand1 >= 260\
        and p_brand1 <= 267\
        and s_region = 2\
        group by d_year,p_brand1;",
    23: "select sum(lo_revenue),d_year,p_brand1\
        from lineorder,part,supplier,ddate\
        where lo_orderdate = d_datekey\
        and lo_partkey = p_partkey\
        and lo_suppkey = s_suppkey\
        and p_brand1 = 260\
        and s_region = 3\
        group by d_year,p_brand1;",
    31: "select c_nation,s_nation,d_year,sum(lo_revenue) as revenue\
        from lineorder,customer, supplier,ddate\
        where lo_custkey = c_custkey\
        and lo_suppkey = s_suppkey\
        and lo_orderdate = d_datekey\
        and c_region = 2\
        and s_region = 2\
        and d_year >= 1992 and d_year <= 1997\
        group by c_nation,s_nation,d_year;",
    32: "select c_city,s_city,d_year,sum(lo_revenue) as revenue\
        from lineorder,customer,supplier,ddate\
        where lo_custkey = c_custkey\
        and lo_suppkey = s_suppkey\
        and lo_orderdate = d_datekey\
        and c_nation = 24\
        and s_nation = 24\
        and d_year >=1992 and d_year <= 1997\
        group by c_city,s_city,d_year;",
    33: "select c_city,s_city,d_year,sum(lo_revenue) as revenue\
        from lineorder,customer,supplier,ddate\
        where lo_custkey = c_custkey\
        and lo_suppkey = s_suppkey\
        and lo_orderdate = d_datekey\
        and (c_city = 231 or c_city = 235)\
        and (s_city = 231 or s_city = 235)\
        and d_year >=1992 and d_year <= 1997\
        group by c_city,s_city,d_year;",
    34: "select c_city,s_city,d_year,sum(lo_revenue) as revenue\
        from lineorder,customer,supplier,ddate\
        where lo_suppkey = s_suppkey\
        and lo_custkey = c_custkey\
        and lo_orderdate = d_datekey\
        and (c_city = 231 or c_city = 235)\
        and (s_city = 231 or s_city = 235)\
        and d_yearmonthnum = 199712\
        group by c_city,s_city,d_year;",
    41: "select d_year,c_nation,sum(lo_revenue-lo_supplycost) as profit\
        from lineorder,supplier,customer,part,ddate\
        where lo_custkey = c_custkey\
        and lo_suppkey = s_suppkey\
        and lo_partkey = p_partkey\
        and lo_orderdate = d_datekey\
        and c_region = 1\
        and s_region = 1\
        and (p_mfgr = 0 or p_mfgr = 1)\
        group by d_year,c_nation;",
    42: "select d_year,s_nation,p_category,sum(lo_revenue-lo_supplycost) as profit\
        from lineorder,customer,supplier,part,ddate\
        where lo_custkey = c_custkey\
        and lo_suppkey = s_suppkey\
        and lo_partkey = p_partkey\
        and lo_orderdate = d_datekey\
        and c_region = 1\
        and s_region = 1\
        and (d_year = 1997 or d_year = 1998)\
        and (p_mfgr = 0 or p_mfgr = 1)\
        group by d_year,s_nation, p_category;",
    43: "select d_year,s_city,p_brand1,sum(lo_revenue-lo_supplycost) as profit\
        from lineorder,supplier,customer,part,ddate\
        where lo_custkey = c_custkey\
        and lo_suppkey = s_suppkey\
        and lo_partkey = p_partkey\
        and lo_orderdate = d_datekey\
        and c_region = 1\
        and s_nation = 24\
        and (d_year = 1997 or d_year = 1998)\
        and p_category = 3\
        group by d_year,s_city,p_brand1;"
}
results = []
with ddb.connect('ssb.db') as conn:
    for name, query in queries.items():
        print(f"Starting Q{name}")
        best_time = float('inf')
        for _ in range(5):
            start_time = time.time()
            result = conn.sql(query).fetchall()
            # use explain to show the query plan
            result = conn.sql(f"explain {query}").fetchall()
            end_time = time.time()
            print(f"Q{name} took: {(end_time - start_time) * 1000}")
            best_time = min(best_time, end_time - start_time)
        print(f"Q{name} best: {best_time * 1000}")
        results.append({
            'query': name,
            'time': best_time * 1000
        })

df_results = pd.DataFrame(results)
# df_results.to_csv('results.csv', index=False)
