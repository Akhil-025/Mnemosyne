"""
Geocoding utilities for Mnemosyne
"""
import logging
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class LocationInfo:
    """Location information"""
    latitude: float
    longitude: float
    altitude: Optional[float] = None
    name: Optional[str] = None
    display_name: Optional[str] = None
    address: Optional[Dict[str, str]] = None
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None
    country_code: Optional[str] = None
    postcode: Optional[str] = None
    timezone: Optional[str] = None
    place_id: Optional[int] = None
    osm_id: Optional[int] = None
    osm_type: Optional[str] = None
    category: Optional[str] = None
    type: Optional[str] = None
    confidence: float = 0.0
    source: str = "unknown"
    timestamp: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            k: v for k, v in self.__dict__.items() 
            if v is not None
        }
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), default=str)
    
    def get_formatted_address(self) -> str:
        """Get formatted address string"""
        if self.display_name:
            return self.display_name
        
        parts = []
        if self.address:
            parts.extend([
                self.address.get('road'),
                self.address.get('suburb'),
                self.address.get('city', self.address.get('town', self.address.get('village'))),
                self.address.get('state'),
                self.address.get('country')
            ])
        else:
            parts.extend([
                self.city,
                self.state,
                self.country
            ])
        
        # Filter out None values and join
        return ', '.join(filter(None, parts))
    
    def get_short_location(self) -> str:
        """Get short location string (City, Country)"""
        if self.city and self.country:
            return f"{self.city}, {self.country}"
        elif self.country:
            return self.country
        elif self.name:
            return self.name
        else:
            return f"{self.latitude:.4f}, {self.longitude:.4f}"


class GeoCoder:
    """Geocoding and reverse geocoding service"""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path.home() / ".mnemosyne" / "cache" / "geocoding"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize providers
        self.nominatim_enabled = False
        self.google_enabled = False
        self.offline_enabled = False
        
        self._init_providers()
        
        # Cache
        self.cache = {}
        self._load_cache()
    
    def _init_providers(self):
        """Initialize geocoding providers"""
        # Try Nominatim (OpenStreetMap) first
        try:
            from geopy.geocoders import Nominatim
            from geopy.exc import GeopyError
            
            self.nominatim = Nominatim(
                user_agent="mnemosyne-photo-organizer/1.0",
                timeout=10
            )
            self.nominatim_enabled = True
            logger.info("Nominatim geocoder initialized")
        except ImportError:
            logger.warning("geopy not installed, Nominatim geocoder disabled")
        except Exception as e:
            logger.warning(f"Failed to initialize Nominatim: {e}")
        
        # Try offline geocoding
        try:
            from timezonefinder import TimezoneFinder
            self.timezone_finder = TimezoneFinder()
            self.offline_enabled = True
            logger.info("Offline timezone finder initialized")
        except ImportError:
            logger.warning("timezonefinder not installed, offline geocoding disabled")
        except Exception as e:
            logger.warning(f"Failed to initialize offline geocoder: {e}")
    
    def _get_cache_key(self, latitude: float, longitude: float) -> str:
        """Generate cache key for coordinates"""
        # Round to 4 decimal places (~11 meter precision)
        lat_key = f"{latitude:.4f}"
        lon_key = f"{longitude:.4f}"
        return f"{lat_key}_{lon_key}"
    
    def _load_cache(self):
        """Load geocoding cache from disk"""
        cache_file = self.cache_dir / "geocoding_cache.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
                logger.info(f"Loaded {len(self.cache)} geocoding cache entries")
            except Exception as e:
                logger.error(f"Error loading geocoding cache: {e}")
                self.cache = {}
        else:
            self.cache = {}
    
    def _save_cache(self):
        """Save geocoding cache to disk"""
        cache_file = self.cache_dir / "geocoding_cache.json"
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving geocoding cache: {e}")
    
    def _get_from_cache(self, latitude: float, longitude: float) -> Optional[LocationInfo]:
        """Get location info from cache"""
        cache_key = self._get_cache_key(latitude, longitude)
        
        if cache_key in self.cache:
            cached_data = self.cache[cache_key]
            
            # Check if cache is still valid (7 days)
            cached_time = datetime.fromisoformat(cached_data.get('timestamp', '2000-01-01'))
            cache_age = (datetime.now() - cached_time).days
            
            if cache_age < 7:
                logger.debug(f"Cache hit for {latitude}, {longitude}")
                return LocationInfo(**cached_data)
        
        return None
    
    def _add_to_cache(self, location: LocationInfo):
        """Add location info to cache"""
        cache_key = self._get_cache_key(location.latitude, location.longitude)
        self.cache[cache_key] = location.to_dict()
        
        # Save cache periodically
        if len(self.cache) % 10 == 0:
            self._save_cache()
    
    async def reverse_geocode(self, latitude: float, longitude: float, 
                             altitude: Optional[float] = None) -> Optional[LocationInfo]:
        """
        Reverse geocode coordinates to location information
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            altitude: Optional altitude
            
        Returns:
            LocationInfo object, or None if error
        """
        # Check cache first
        cached = self._get_from_cache(latitude, longitude)
        if cached:
            if altitude is not None:
                cached.altitude = altitude
            return cached
        
        location_info = None
        
        # Try Nominatim first
        if self.nominatim_enabled:
            location_info = await self._reverse_geocode_nominatim(latitude, longitude)
        
        # If Nominatim failed or not available, use offline method
        if not location_info and self.offline_enabled:
            location_info = await self._reverse_geocode_offline(latitude, longitude)
        
        if location_info:
            location_info.altitude = altitude
            location_info.timestamp = datetime.now().isoformat()
            
            # Add to cache
            self._add_to_cache(location_info)
            
            logger.info(f"Reverse geocoded {latitude}, {longitude} -> {location_info.get_short_location()}")
            return location_info
        
        logger.warning(f"Failed to reverse geocode {latitude}, {longitude}")
        return None
    
    async def _reverse_geocode_nominatim(self, latitude: float, longitude: float) -> Optional[LocationInfo]:
        """Reverse geocode using Nominatim (OpenStreetMap)"""
        try:
            import asyncio
            from geopy.exc import GeopyError
            
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            
            location = await loop.run_in_executor(
                None,
                lambda: self.nominatim.reverse((latitude, longitude), exactly_one=True, language='en')
            )
            
            if not location:
                return None
            
            raw_data = location.raw if hasattr(location, 'raw') else {}
            address = raw_data.get('address', {})
            
            return LocationInfo(
                latitude=latitude,
                longitude=longitude,
                name=location.address,
                display_name=location.address,
                address=address,
                city=address.get('city') or address.get('town') or address.get('village'),
                state=address.get('state'),
                country=address.get('country'),
                country_code=address.get('country_code'),
                postcode=address.get('postcode'),
                place_id=raw_data.get('place_id'),
                osm_id=raw_data.get('osm_id'),
                osm_type=raw_data.get('osm_type'),
                category=raw_data.get('category'),
                type=raw_data.get('type'),
                confidence=0.8,
                source="nominatim"
            )
            
        except Exception as e:
            logger.debug(f"Nominatim reverse geocoding failed: {e}")
            return None
    
    async def _reverse_geocode_offline(self, latitude: float, longitude: float) -> Optional[LocationInfo]:
        """Reverse geocode using offline methods"""
        try:
            # Get timezone
            timezone = self.get_timezone(latitude, longitude)
            
            # Simple location approximation
            # In production, you'd use an offline geocoding database
            location_info = LocationInfo(
                latitude=latitude,
                longitude=longitude,
                timezone=timezone,
                confidence=0.5,
                source="offline"
            )
            
            return location_info
            
        except Exception as e:
            logger.debug(f"Offline reverse geocoding failed: {e}")
            return None
    
    def get_timezone(self, latitude: float, longitude: float) -> Optional[str]:
        """
        Get timezone for coordinates
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            
        Returns:
            Timezone string (e.g., 'America/New_York'), or None if error
        """
        if not self.offline_enabled:
            return None
        
        try:
            timezone = self.timezone_finder.timezone_at(lat=latitude, lng=longitude)
            return timezone
        except Exception as e:
            logger.debug(f"Failed to get timezone for {latitude}, {longitude}: {e}")
            return None
    
    async def geocode(self, location_name: str) -> List[LocationInfo]:
        """
        Geocode location name to coordinates
        
        Args:
            location_name: Name of location (e.g., "Paris, France")
            
        Returns:
            List of LocationInfo objects
        """
        if not self.nominatim_enabled:
            logger.warning("Nominatim not available for geocoding")
            return []
        
        try:
            import asyncio
            
            loop = asyncio.get_event_loop()
            
            locations = await loop.run_in_executor(
                None,
                lambda: self.nominatim.geocode(location_name, exactly_one=False, language='en')
            )
            
            if not locations:
                return []
            
            results = []
            for location in locations[:5]:  # Limit to 5 results
                raw_data = location.raw if hasattr(location, 'raw') else {}
                address = raw_data.get('address', {})
                
                location_info = LocationInfo(
                    latitude=location.latitude,
                    longitude=location.longitude,
                    name=location.address,
                    display_name=location.address,
                    address=address,
                    city=address.get('city') or address.get('town') or address.get('village'),
                    state=address.get('state'),
                    country=address.get('country'),
                    country_code=address.get('country_code'),
                    postcode=address.get('postcode'),
                    place_id=raw_data.get('place_id'),
                    osm_id=raw_data.get('osm_id'),
                    osm_type=raw_data.get('osm_type'),
                    category=raw_data.get('category'),
                    type=raw_data.get('type'),
                    confidence=0.8,
                    source="nominatim"
                )
                
                results.append(location_info)
            
            logger.info(f"Geocoded '{location_name}' to {len(results)} locations")
            return results
            
        except Exception as e:
            logger.error(f"Error geocoding '{location_name}': {e}")
            return []
    
    def calculate_distance(self, lat1: float, lon1: float, 
                          lat2: float, lon2: float) -> float:
        """
        Calculate distance between two coordinates in kilometers
        
        Args:
            lat1, lon1: First coordinates
            lat2, lon2: Second coordinates
            
        Returns:
            Distance in kilometers
        """
        try:
            from math import radians, sin, cos, sqrt, atan2
            
            # Convert to radians
            lat1_rad = radians(lat1)
            lon1_rad = radians(lon1)
            lat2_rad = radians(lat2)
            lon2_rad = radians(lon2)
            
            # Haversine formula
            dlon = lon2_rad - lon1_rad
            dlat = lat2_rad - lat1_rad
            
            a = sin(dlat/2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon/2)**2
            c = 2 * atan2(sqrt(a), sqrt(1-a))
            
            # Earth radius in kilometers
            radius = 6371.0
            
            return radius * c
            
        except Exception as e:
            logger.error(f"Error calculating distance: {e}")
            return 0.0
    
    def find_nearby_locations(self, center_lat: float, center_lon: float,
                             locations: List[LocationInfo],
                             radius_km: float = 10.0) -> List[LocationInfo]:
        """
        Find locations within radius of center coordinates
        
        Args:
            center_lat, center_lon: Center coordinates
            locations: List of locations to search
            radius_km: Search radius in kilometers
            
        Returns:
            List of locations within radius
        """
        nearby = []
        
        for location in locations:
            distance = self.calculate_distance(
                center_lat, center_lon,
                location.latitude, location.longitude
            )
            
            if distance <= radius_km:
                location_with_distance = location
                location_with_distance.confidence = min(1.0, 1.0 - (distance / radius_km))
                nearby.append(location_with_distance)
        
        # Sort by distance (closest first)
        nearby.sort(key=lambda loc: self.calculate_distance(
            center_lat, center_lon, loc.latitude, loc.longitude
        ))
        
        return nearby


# Global geocoder instance
_geocoder_instance = None

def get_geocoder() -> GeoCoder:
    """Get global geocoder instance"""
    global _geocoder_instance
    if _geocoder_instance is None:
        _geocoder_instance = GeoCoder()
    return _geocoder_instance


async def reverse_geocode(latitude: float, longitude: float, 
                         altitude: Optional[float] = None) -> Optional[LocationInfo]:
    """Reverse geocode coordinates using global geocoder"""
    geocoder = get_geocoder()
    return await geocoder.reverse_geocode(latitude, longitude, altitude)


def get_timezone(latitude: float, longitude: float) -> Optional[str]:
    """Get timezone for coordinates using global geocoder"""
    geocoder = get_geocoder()
    return geocoder.get_timezone(latitude, longitude)


async def geocode_location(location_name: str) -> List[LocationInfo]:
    """Geocode location name using global geocoder"""
    geocoder = get_geocoder()
    return await geocoder.geocode(location_name)


def calculate_distance(lat1: float, lon1: float, 
                      lat2: float, lon2: float) -> float:
    """Calculate distance between coordinates"""
    geocoder = get_geocoder()
    return geocoder.calculate_distance(lat1, lon1, lat2, lon2)