import osmnx as ox
import streamlit as st
import pandas as pd
import folium
import logging
import json
from rich.logging import RichHandler
from streamlit_folium import folium_static

# Configure logging for internal debugging using Rich.
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()]
)
logger = logging.getLogger("rich")

def calculate_static_street_view_cost(num_requests):
    """
    Calculate the total cost for the given number of requests based on pricing tiers.
    A free cap of 10,000 requests is applied.
    """
    tiers = [
        (100_000, 7.00),         # 0 to 100,000 requests: USD$7.00 per 1,000
        (400_000, 5.60),         # 100,001 to 500,000 requests: USD$5.60 per 1,000
        (500_000, 4.20),         # 500,001 to 1,000,000 requests: USD$4.20 per 1,000
        (4_000_000, 2.10),       # 1,000,001 to 5,000,000 requests: USD$2.10 per 1,000
        (float('inf'), 0.53)     # 5,000,001+ requests: USD$0.53 per 1,000
    ]
    free_cap = 10_000
    if num_requests <= free_cap:
        return 0.0

    num_requests -= free_cap
    cost = 0.0
    lower_bound = 0

    for tier_limit, price_per_thousand in tiers:
        tier_range = min(num_requests, tier_limit - lower_bound)
        if tier_range <= 0:
            break
        cost += (tier_range / 1000) * price_per_thousand
        num_requests -= tier_range
        lower_bound = tier_limit

    return round(cost, 2)

@st.cache_data(show_spinner=True)
def get_building_count(city_name):
    """
    Geocode the city, query OSM for building footprints within its boundary,
    and return the total building count.
    """
    try:
        logger.info("Fetching data for %s from OSM...", city_name)
        gdf = ox.geocode_to_gdf(city_name)
        if hasattr(gdf.geometry, "union_all"):
            boundary_polygon = gdf.geometry.union_all()
        else:
            boundary_polygon = gdf.unary_union
        tags = {"building": True}
        buildings = ox.features.features_from_polygon(boundary_polygon, tags)
        count = len(buildings)
        logger.info("Retrieved %d buildings for %s.", count, city_name)
        return count
    except Exception as e:
        logger.exception("Error fetching data for %s: %s", city_name, e)
        return None

@st.cache_data(show_spinner=True)
def get_building_footprints(city_name):
    """
    Geocode the city, query OSM for building footprints within its boundary,
    and return a GeoDataFrame of building footprints.
    """
    try:
        logger.info("Fetching building footprints for %s from OSM...", city_name)
        gdf = ox.geocode_to_gdf(city_name)
        if hasattr(gdf.geometry, "union_all"):
            boundary_polygon = gdf.geometry.union_all()
        else:
            boundary_polygon = gdf.unary_union
        tags = {"building": True}
        buildings = ox.features.features_from_polygon(boundary_polygon, tags)
        logger.info("Retrieved %d building footprints for %s.", len(buildings), city_name)
        return buildings
    except Exception as e:
        logger.exception("Error fetching building footprints for %s: %s", city_name, e)
        return None

@st.cache_data(show_spinner=True)
def get_city_boundary(city_name):
    """
    Geocode the city and return its boundary as a (Multi)Polygon.
    """
    try:
        logger.info("Fetching boundary for %s from OSM...", city_name)
        gdf = ox.geocode_to_gdf(city_name)
        if hasattr(gdf.geometry, "union_all"):
            boundary_polygon = gdf.geometry.union_all()
        else:
            boundary_polygon = gdf.unary_union
        return boundary_polygon
    except Exception as e:
        logger.exception("Error fetching boundary for %s: %s", city_name, e)
        return None

def create_boundary_map(city_name, plot_buildings=False):
    """
    Create a Folium map displaying the boundary of the given city.
    Optionally, overlay building footprints unless there are more than 5000.
    """
    boundary_polygon = get_city_boundary(city_name)
    if boundary_polygon is None:
        return None
    centroid = [boundary_polygon.centroid.y, boundary_polygon.centroid.x]
    m = folium.Map(location=centroid, zoom_start=12)
    folium.GeoJson(boundary_polygon, name=f"{city_name} Boundary").add_to(m)
    if plot_buildings:
        buildings = get_building_footprints(city_name)
        if buildings is not None and not buildings.empty:
            if len(buildings) > 5000:
                logger.info("Too many building footprints (%d) for %s; not plotting.",
                            len(buildings), city_name)
            else:
                folium.GeoJson(data=buildings.__geo_interface__, name="Building Footprints").add_to(m)
    folium.LayerControl().add_to(m)
    return m

def find_sampling_percentage(total_buildings, target_cost, tol=0.01):
    """
    Given the total building count and a target cost, use binary search to find
    the sampling percentage (0 to 100) that approximately yields the target cost.
    """
    max_cost = calculate_static_street_view_cost(total_buildings)
    if target_cost > max_cost:
        return None

    lo, hi = 0.0, 100.0
    while hi - lo > tol:
        mid = (lo + hi) / 2
        current_cost = calculate_static_street_view_cost(total_buildings * (mid / 100))
        if current_cost < target_cost:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2

def display_results_table(results, overall_cost, sampling_percentage, mode):
    """
    Display a table of results using Streamlit's table functionality.
    """
    df = pd.DataFrame(results)
    st.subheader("Results per City")
    st.table(df)
    st.subheader("Overall Results")
    if mode == "Cost from Sampling Percentage":
        st.write(f"Sampling Percentage: **{sampling_percentage:.2f}%**")
        st.write(f"Overall Cost (from aggregated sample): **USD${overall_cost:,.2f}**")
    else:
        st.write(f"Computed Sampling Percentage: **{sampling_percentage:.2f}%**")
        st.write(f"Overall Cost at this sampling: **USD${overall_cost:,.2f}**")

def main():
    st.title("Street View Cost & Sampling Calculator")
    st.markdown(
        """
        This app calculates the cost to capture a street view image for a percentage of buildings
        in a list of cities OR computes the sampling percentage required to reach a target cost.
        It also lets you view a map of each city's boundary and optionally overlay building footprints.
        Note: The overall cost is calculated from the total number of buildings sampled across cities,
        applying the free quota and tiered discounts.
        """
    )

    mode = st.radio("Calculation Mode", ("Cost from Sampling Percentage", "Sampling Percentage for Target Cost"))

    # Updated default cities.
    cities_input = st.text_area("Cities (one per line)",
                                "Little Portugal, Toronto, Ontario, Canada\nChinatown, Calgary, Alberta, Canada",
                                height=150)
    show_maps = st.checkbox("Show Boundary Maps", value=False)
    plot_buildings = st.checkbox("Plot Building Footprints on Map", value=False)

    cities = [city.strip() for city in cities_input.splitlines() if city.strip()]
    if not cities:
        st.error("Please enter at least one city.")
        return

    if mode == "Cost from Sampling Percentage":
        sampling_percentage = st.slider("Sampling Percentage (%)", 0, 100, 10)
    else:
        target_cost = st.number_input("Target Total Cost (USD$)", min_value=0.0, value=1000.0, step=10.0)

    if st.button("Calculate"):
        results = []
        overall_buildings = 0

        progress_bar = st.progress(0)
        progress_text = st.empty()

        for idx, city in enumerate(cities, start=1):
            progress_text.text(f"Processing {city} ({idx} of {len(cities)})...")
            building_count = get_building_count(city)
            if building_count is None:
                st.error(f"Failed to fetch data for {city}. Please verify the city name.")
                continue
            overall_buildings += building_count
            results.append({
                "City": city,
                "Building Count": building_count
            })
            progress_bar.progress(idx / len(cities))
        progress_text.empty()

        if overall_buildings == 0:
            st.error("No building data was successfully retrieved.")
            return

        # In both modes, overall cost is computed from the aggregated building count.
        if mode == "Cost from Sampling Percentage":
            for result in results:
                building_count = result["Building Count"]
                num_requests = int(building_count * (sampling_percentage / 100))
                cost = calculate_static_street_view_cost(num_requests)
                result["Requests"] = num_requests
                result["Cost (USD$)"] = cost
            overall_cost = calculate_static_street_view_cost(overall_buildings * (sampling_percentage / 100))
            display_results_table(results, overall_cost, sampling_percentage, mode)
            export_sampling = sampling_percentage
        else:
            computed_sampling = find_sampling_percentage(overall_buildings, target_cost)
            if computed_sampling is None:
                st.error("Target cost exceeds the maximum cost at 100% sampling for the given cities.")
                return
            overall_cost = calculate_static_street_view_cost(overall_buildings * (computed_sampling / 100))
            for result in results:
                building_count = result["Building Count"]
                num_requests = int(building_count * (computed_sampling / 100))
                cost = calculate_static_street_view_cost(num_requests)
                result["Requests"] = num_requests
                result["Cost (USD$)"] = cost
            display_results_table(results, overall_cost, computed_sampling, mode)
            export_sampling = computed_sampling

        if show_maps:
            st.subheader("Boundary Maps")
            for city in cities:
                st.markdown(f"**{city}**")
                m = create_boundary_map(city, plot_buildings=plot_buildings)
                if m is not None:
                    folium_static(m)
                else:
                    st.error(f"Could not generate a map for {city}.")

        # Prepare export data: boundaries for each city and the sampling percentage.
        export_data = {
            "sampling_percentage": export_sampling,
            "regions": []
        }
        for city in cities:
            boundary = get_city_boundary(city)
            if boundary is not None:
                export_data["regions"].append({
                    "city": city,
                    "boundary": boundary.__geo_interface__
                })
        # Convert export data to JSON.
        export_json = json.dumps(export_data, indent=2)
        st.download_button(
            label="Export Data",
            data=export_json,
            file_name="export.json",
            mime="application/json"
        )

if __name__ == "__main__":
    main()

