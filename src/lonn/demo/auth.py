"""Small password verifier for local demo users, not a registration service."""
import hashlib
import secrets


def password_hash(password):
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 210_000).hex()
    return f'pbkdf2_sha256${salt}${digest}'


def verify_password(password, encoded):
    _, salt, expected = encoded.split('$')
    actual = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 210_000).hex()
    return secrets.compare_digest(actual, expected)
