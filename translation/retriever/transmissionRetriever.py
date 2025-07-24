import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
import pandas as pd
import requests
import geopandas as gpd
import pycountry
import overpass
import logging
from tqdm import tqdm
import folium
from folium.plugins import MarkerCluster
import time
class transmissionRetrieverClass():
    def __init__(self, logger, regions):
        self.logger = logger
        self.regions = regions
        

    def geom_to_overpass_poly(self, geom):
        """Convert a Polygon or MultiPolygon to Overpass poly string."""
        if isinstance(geom, Polygon):
            polygons = [geom]
        elif isinstance(geom, MultiPolygon):
            polygons = list(geom)
        else:
            raise ValueError("Geometry must be Polygon or MultiPolygon")

        poly_strings = []
        for poly in polygons:
            coords = poly.exterior.coords
            coord_str = " ".join(f"{y} {x}" for x, y in coords)  # lat lon
            poly_strings.append(coord_str)

        return poly_strings

    def parse_osm_lines(self, osm_json):
        """Convert Overpass OSM response into a GeoDataFrame of LineStrings with tags."""
        elements = osm_json.get('elements', [])
        nodes = {el['id']: (el['lon'], el['lat']) for el in elements if el['type'] == 'node'}
        lines = []

        for el in elements:
            if el['type'] == 'way' and 'nodes' in el:
                coords = [nodes[nid] for nid in el['nodes'] if nid in nodes]
                if len(coords) < 2:
                    continue
                line = LineString(coords)
                tags = el.get('tags', {})
                tags['geometry'] = line
                lines.append(tags)
        
        if not lines:
            return gpd.GeoDataFrame(columns=['geometry'], geometry='geometry', crs="EPSG:4326")

        return gpd.GeoDataFrame(lines, geometry='geometry', crs="EPSG:4326")

    def get_power_lines(self):
        self.logger.info("Querying Overpass API for power lines")
        countries = self.get_borders(self.regions)

        # Sum all geometries in the countries GeoDataFrame into a single MultiPolygon
        if len(countries) > 1:
            merged_geom = countries.unary_union
        else:
            merged_geom = countries.geometry.iloc[0]

        poly_coords = self.geom_to_overpass_poly(merged_geom)
        query_parts = []
        for poly in poly_coords:
            query_parts.append(f"""
                way["power"="line"](poly:"{poly}");
                relation["power"="line"](poly:"{poly}");
            """)

        overpass_query = f"""
            [out:json][timeout:180];
            (
                {''.join(query_parts)}
            );
            out body;
            >;
            out skel qt;
        """

        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = requests.post("https://overpass-api.de/api/interpreter", data={'data': overpass_query})
                response.raise_for_status()
                osm_data = response.json()
                gdf = self.parse_osm_lines(osm_data)
                self.logger.info(f"Retrieved {len(gdf)} power lines for all countries")
                break
            except Exception as e:
                self.logger.error(f"Attempt {attempt+1}/{max_retries} - Error retrieving lines: {e}")
                time.sleep(5)
                if attempt == max_retries - 1:
                    raise Exception(f"Failed to retrieve power lines after {max_retries} attempts: {e}")
        if gdf.empty:
            self.logger.warning("No power lines found in the specified regions.")
            return countries, gdf

        result = gdf.reset_index(drop=True)
        self.logger.info(f"Total retrieved power lines: {len(result)}")
        return countries, result

    def deduplicate_cross_border_lines(self, df, tolerance=1e-6):
        # Normalize country pairs to treat (ZW, ZM) and (ZM, ZW) as duplicates
        df['country_pair'] = df.apply(
            lambda row: tuple(sorted([row['start_country'], row['end_country']])), axis=1
        )

        # We'll store indices of rows to keep
        keep_indices = []
        seen = []
        for idx, row in df.iterrows():
            duplicate_found = False
            for prev in seen:
                same_pair = row['country_pair'] == prev['country_pair']
                same_voltage = row['voltage'] == prev['voltage']
                same_circuit = row['circuit'] == prev['circuit']
                same_cables = row['cables'] == prev['cables']
                same_geom = row.geometry.equals_exact(prev['geometry'], tolerance)

                if same_pair and same_voltage and same_circuit and same_cables and same_geom:
                    duplicate_found = True
                    break

            if not duplicate_found:
                keep_indices.append(idx)
                seen.append(row)

        return df.loc[keep_indices].drop(columns=['country_pair']).reset_index(drop=True)

    def get_borders(self, countries_array):
        # Handle NM special case: replace NM with NA for shapefile lookup, then revert to NM
        countries_array_fixed = ["NA" if c == "NM" else c for c in countries_array]
        world = gpd.read_file("./data/shapefiles/ne_110m_admin_0_countries.shp")
        filtered_countries = world[world['ISO_A2'].isin(countries_array_fixed)].copy()
        filtered_countries['ISO_A2'] = filtered_countries['ISO_A2'].replace("NA", "NM")
        return filtered_countries
    
    def get_line_countries(self, line, countries):
        start = Point(line.coords[0])
        end = Point(line.coords[-1])
        
        start_country = countries[countries.contains(start)]
        end_country = countries[countries.contains(end)]

        if start_country.empty or end_country.empty:
            return None, None  # skip if unknown
        return start_country.iloc[0]['ISO_A2'], end_country.iloc[0]['ISO_A2']
    
    def estimate_capacity_per_circuit(self, voltage_kv):
        """
        Estimate thermal transmission line capacity per circuit based on voltage level (in kV).
        
        Parameters:
        - voltage_kv (float or int): Nominal voltage of the line in kilovolts (kV)

        Returns:
        - capacity: Tuple with estimated capacity range in MWh
        """
        if voltage_kv < 60:
            return 50
        elif 60 <= voltage_kv < 120:
            return 200
        elif 120 <= voltage_kv <= 160:
            return 400
        elif 160 < voltage_kv <= 250:
            return 700
        elif 250 < voltage_kv <= 350:
            return 1000
        elif 350 < voltage_kv <= 450:
            return 1600
        elif 450 < voltage_kv <= 550:
            return 2200
        else:  # HVDC or very high-voltage AC
            return 2500
    
    def extract_cross_border_lines(self, unit='TJ', yearly=False):
        self.logger.info("Extracting cross-border lines")

        countries, lines = self.get_power_lines()
        cross_border_lines = []
        for idx, line in tqdm(lines.iterrows(), desc="Processing transmission lines", total=len(lines)):
            start_country, end_country = self.get_line_countries(line.geometry, countries)
            if start_country != end_country:
                tension = line.get('voltage', None)
                circuit = line.get('circuits', None)
                cables = line.get('cables', None)
                if pd.notna(start_country) and pd.notna(end_country):
                    cross_border_lines.append({
                        "start_country": start_country,
                        "end_country": end_country,
                        "voltage": int(tension) if pd.notna(tension) else None,
                        "circuit": int(circuit) if pd.notna(circuit) else None,
                        "cables": int(cables) if pd.notna(cables) else None,
                        "geometry": line.geometry,
                    })
        df = pd.DataFrame(cross_border_lines)
        self.logger.info(f"Extracted {len(df)} cross-border lines")
        df['capacity'] = df['voltage'].fillna(150).apply(self.estimate_capacity_per_circuit)*df['circuit'].fillna(1)  # MWh per year
        #df = self.deduplicate_cross_border_lines(df, tolerance=1e-6)

        df = df.groupby(['start_country', 'end_country'], as_index=False).agg({
            'capacity': 'sum',
            'geometry': 'first',  # Keep the first geometry for simplicity
            'voltage': 'first',  # Keep the first voltage for simplicity
            'circuit': 'first',  # Keep the first circuit for simplicity
            'cables': 'first'    # Keep the first cables for simplicity
        })
        df = df[['start_country', 'end_country', 'capacity']].reset_index(drop=True)
        df_copy = df.copy()
        df_copy = df_copy.rename(columns={"start_country": "end_country", "end_country": "start_country"})
        df = pd.concat([df, df_copy], ignore_index=True).groupby(['start_country', 'end_country'], as_index=False).agg({
            'capacity': 'sum'
        })
        if unit == 'GWh':
            df['capacity'] = df['capacity'] * 1e-3
        elif unit == 'TJ':
            df['capacity'] = df['capacity'] * 1e-3 * 3.6
        else:
            raise ValueError("Unsupported unit. Use 'GWh' or 'TJ'.")
        if yearly:
            df['capacity'] = df['capacity'] * 8760
        return df
    

if __name__ == "__main__":
    # Example usage
    logger = logging.getLogger(__name__)
    regions = ["AO", "BW", "CD", "LS", "MW", "MZ", "NA", "SZ", "TZ", "ZA", "ZM", "ZW"]  # Updated country codes
    retriever = transmissionRetrieverClass(logger, regions)
    cross_border_lines = retriever.extract_cross_border_lines()
    print(cross_border_lines)