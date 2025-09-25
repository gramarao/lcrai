from dotenv import load_dotenv
import os
load_dotenv()
print(os.getenv("DB_PASSWORD"))  # Should match psql password
