import psycopg2

# Connect to both databases
local_conn = psycopg2.connect("postgresql://localhost/attendance_system")
railway_conn = psycopg2.connect("postgresql://postgres:NraeRTIAGwBMQoAJXbzJhmqKtSwVxYCQ@centerbeam.proxy.rlwy.net:52150/railway")

# Example: Copy one table
with local_conn.cursor() as local_cur, railway_conn.cursor() as railway_cur:
    local_cur.execute("SELECT * FROM students")
    for row in local_cur:
        railway_cur.execute("INSERT INTO students VALUES (%s, %s, %s)", row)
    railway_conn.commit()