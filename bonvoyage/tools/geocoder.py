import time
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError


_geolocator = Nominatim(user_agent="bonvoyage-travel-agent")


def geocode_address(address: str, city: str) -> tuple[float | None, float | None]:
    """Return (lat, lon) for address. Returns (None, None) on failure."""
    if not address:
        return None, None

    query = f"{address}, {city}" if city.lower() not in address.lower() else address
    try:
        time.sleep(1)  # Nominatim requires 1 req/sec
        location = _geolocator.geocode(query, timeout=5)
        if location:
            return location.latitude, location.longitude
    except (GeocoderTimedOut, GeocoderServiceError):
        pass

    # Retry with city-only context if full address fails
    try:
        time.sleep(1)
        location = _geolocator.geocode(f"{address.split(',')[0]}, {city}", timeout=5)
        if location:
            return location.latitude, location.longitude
    except (GeocoderTimedOut, GeocoderServiceError):
        pass

    return None, None
