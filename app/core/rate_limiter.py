from slowapi import Limiter
from slowapi.util import get_remote_address

# Configure rate limiter resolving client by IP address
limiter = Limiter(key_func=get_remote_address)
