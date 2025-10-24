INSERT INTO users (username, email, hashed_password, full_name, role, is_active)
VALUES (
    'admin',
    'gramarao@gmail.com',
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5aeUZ.3K5.6WK',  -- admin123
    'System Admin',
    'superuser',
    true
);