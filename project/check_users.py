"""
Check what's actually in your users.db - run this if login/register
seems broken, to see the real state of the database.

Run:
    python check_users.py
"""
import sqlite3
import os

if not os.path.exists("users.db"):
    print("users.db does not exist yet - it will be created when you run app.py")
else:
    conn = sqlite3.connect("users.db")
    conn.row_factory = sqlite3.Row

    cols = conn.execute("PRAGMA table_info(users)").fetchall()
    col_names = [c["name"] for c in cols]
    print("Columns in 'users' table:", col_names)

    if "email" not in col_names:
        print("\n⚠️  This is an OLD database (no 'email' column).")
        print("   Delete users.db and restart app.py to fix this.")
    else:
        rows = conn.execute("SELECT id, email FROM users").fetchall()
        print(f"\n{len(rows)} user(s) registered:")
        for r in rows:
            print(f"  - {r['email']}")

    conn.close()
