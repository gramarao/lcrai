#!/usr/bin/env python3
import bcrypt
import sys

def generate_hash(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

if __name__ == "__main__":
    password = sys.argv[1] if len(sys.argv) > 1 else input("Enter password: ")
    hashed = generate_hash(password)
    print(f"\nPassword: {password}")
    print(f"Hashed: {hashed}\n")
