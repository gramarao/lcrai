from app.models.database import SessionLocal, Document
from config.settings import settings

# Print DATABASE_URL for debugging
print(f"Using DATABASE_URL: {settings.DATABASE_URL}")

db = SessionLocal()
try:
    count = db.query(Document).count()
    print(f"Documents: {count}")
except Exception as e:
    print(f"Error: {str(e)}")
finally:
    db.close()
