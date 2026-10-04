from impala.util import as_pandas
from neo4j import GraphDatabase
from dotenv import load_dotenv

from tqdm import tqdm
from decimal import Decimal
import pandas as pd
import os
import numpy as np
from typing import List, Dict, Any
from config import config

load_dotenv()

# Note:
# Managed transactions balance connection management and performance.
# Sessions send batches while the driver handles retries, rollback, and commit.
# The data window is shifted by two months.

# This file preserves the original enterprise ETL template.
# Complete the SQL and connection before enabling it; use load_synthetic_neo4j.py for the demo.
raise NotImplementedError(
    "Configure the Impala ETL: queries and driver are not included. Use make pipeline-synthetic for the demo."
)

# Driver
## Neo4j
driver = config.neo4j_driver
database = os.getenv("NEO4J_DATABASE")

## Impala
connection, cursor = "DA AGGIUNGERE UN DRIVER"

## Parameters
N_CLIENTI = None
N_MINIMO_CONNESSIONI = 4
N_MESI = 12


# ============================================================
# SQL EXTRACTION
# ============================================================


def _base_filter_query(n_clienti, n_minimo_connessioni, n_mesi):
    """
    Base extraction query, providing the initial filter for dimension tables.
    """
    limit_clause = "" if n_clienti is None else f"LIMIT {n_clienti}"

    return f"""

    """


def _query_customer(n_clienti, n_minimo_connessioni, n_mesi):
    return f"""
    {_base_filter_query(n_clienti, n_minimo_connessioni, n_mesi)}

    """


def _query_salesrep(n_clienti, n_minimo_connessioni, n_mesi):
    return f"""
    {_base_filter_query(n_clienti, n_minimo_connessioni, n_mesi)}

    """


def _query_product(n_clienti, n_minimo_connessioni, n_mesi):
    return f"""
    {_base_filter_query(n_clienti, n_minimo_connessioni, n_mesi)}

    """


# Fact tables
def _query_customer_salesrep(n_clienti, n_minimo_connessioni, n_mesi):
    return f"""
    {_base_filter_query(n_clienti, n_minimo_connessioni, n_mesi)}

    """


def _query_customer_product(n_clienti, n_minimo_connessioni, n_mesi):
    return f"""
    {_base_filter_query(n_clienti, n_minimo_connessioni, n_mesi)}

    """


# ============================================================
# TRANSFORMATIONS
# ============================================================


def is_valid(value):
    if value is None:
        return False
    value_str = str(value).strip()
    return value_str not in ("", "NULL", "nan", "NaN", "None")


def normalize(value):
    return str(value).strip() if is_valid(value) else None


def safe_value(value):
    if not is_valid(value):
        return None
    if isinstance(value, Decimal):
        return float(value)
    return value


def normalize_upper(value):
    value = normalize(value)
    return value.upper() if value is not None else None


def clean_scalar(v):
    if pd.isna(v):
        return None
    if isinstance(v, Decimal):
        return float(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, str):
        v = v.strip()
        return v if v != "" else None
    return v


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(lambda col: col.map(clean_scalar))


def transform_customer(row) -> Dict[str, Any] | None:
    customer_id = normalize(row.ids_customer)
    if customer_id is None:
        return None
    return {
        "customerId": customer_id,
        "customer_name": safe_value(row.dsc_customer),
        "customer_partitaIva": safe_value(row.partita_iva),
        "customer_numberEmployees": safe_value(row.att_numberemployees),
        "customer_potential": safe_value(row.att_potential),
        "customer_creditLimit": safe_value(row.att_creditlimit),
        "customer_payTerms": safe_value(row.att_payterms),
        "customer_payTermsDescription": safe_value(row.att_paytermsdescription),
        "customer_paymentMethod": safe_value(row.att_paymentmethod),
        "customer_paymentMethodDescription": safe_value(row.att_paymentmethoddescription),
        "customer_email": safe_value(row.att_email),
        "customer_street": safe_value(row.att_street),
        "customer_zipcode": normalize(row.att_zipcode),
        "customer_longitude": safe_value(row.att_longitude_x),
        "customer_latitude": safe_value(row.att_latitude_y),
        "sectorId": normalize(row.att_sector),
        "sectorDescription": safe_value(row.att_sectordescription),
        "branchId": normalize(row.att_branch),
        "branchDescription": safe_value(row.att_branchdescription),
        "channelId": normalize(row.att_saleschannel),
        "channelDescription": safe_value(row.dsc_channel),
        "marketSegmentId": normalize(row.att_marketsegment),
        "marketSegmentDescription": safe_value(row.att_marketsegmentdescription),
        "cityName": normalize(row.att_city),
        "countyName": normalize(row.att_county),
        "countryName": normalize_upper(row.att_country),
        "salesrepId": normalize(row.ids_salesrep),
    }


def transform_salesrep(row) -> Dict[str, Any] | None:
    salesrep_id = normalize(row.ids_salesrep)
    if salesrep_id is None:
        return None
    return {
        "salesrepId": salesrep_id,
        "salesrep_firstname": safe_value(row.dsc_salesrepfirstname),
        "salesrep_surname": safe_value(row.dsc_salesrepsurname),
        "salesrep_mobile": safe_value(row.att_mobilephonenumber),
        "salesrep_email": safe_value(row.att_email),
        "salesrep_street": safe_value(row.att_street),
        "salesrep_zipcode": normalize(row.att_zipcode),
        "salesrep_city": normalize(row.att_cityname),
        "customerId": normalize(row.ids_customer),
        "channelDescription": safe_value(row.dsc_channel),
        "sectorDescription": safe_value(row.dsc_sector),
        "marketSegmentId": normalize(row.att_marketsegment),
        "businessArea": safe_value(row.att_businessarea),
        "jobDescription": safe_value(row.dsc_job),
        "districtId": normalize(row.ids_district),
        "districtName": safe_value(row.dsc_district),
        "districtSalesrepId": normalize(row.ids_districtsalesrep),
        "districtSalesrepFirstname": safe_value(row.dsc_districtsalesrepfirstname),
        "districtSalesrepSurname": safe_value(row.dsc_districtsalesrepsurname),
        "territoryId": normalize(row.ids_territory),
        "territoryName": safe_value(row.dsc_territory),
        "territorySalesrepId": normalize(row.ids_territorysalesrep),
        "territorySalesrepFirstname": safe_value(row.dsc_territorysalesrepfirstname),
        "territorySalesrepSurname": safe_value(row.dsc_territorysalesrepsurname),
        "mregionId": normalize(row.ids_mregion),
        "mregionName": safe_value(row.dsc_mregion),
        "mregionSalesrepId": normalize(row.ids_mregionsalesrep),
        "mregionSalesrepFirstname": safe_value(row.dsc_mregionsalesrepfirstname),
        "mregionSalesrepSurname": safe_value(row.dsc_mregionsalesrepsurname),
    }


def transform_product(row) -> Dict[str, Any] | None:
    level5_id = normalize(row.level5id)
    if level5_id is None:
        return None
    return {
        "level5Id": level5_id,
        "level5_name": safe_value(row.level5),
        "level4Id": normalize(row.level4id),
        "level4_name": safe_value(row.level4),
        "level3Id": normalize(row.level3id),
        "level3_name": safe_value(row.level3),
        "level2Id": normalize(row.level2id),
        "level2_name": safe_value(row.level2),
        "level1Id": normalize(row.level1id),
        "level1_name": safe_value(row.level1),
    }


def transform_customer_salesrep(row) -> Dict[str, Any] | None:
    customer_id = normalize(row.ids_customer)
    salesrep_id = normalize(row.ids_salesrep)
    if customer_id is None or salesrep_id is None:
        return None
    return {
        "customerId": customer_id,
        "salesrepId": salesrep_id,
    }


def transform_customer_product(row) -> Dict[str, Any] | None:
    level5_id = normalize(row.level5id)
    customer_id = normalize(row.ids_customer)
    if customer_id is None or level5_id is None:
        return None
    return {
        "level5Id": level5_id,
        "customerId": customer_id,
        "dat": safe_value(row.dat),
    }


def extract_and_transform(cursor, query: str, transform_fn) -> List[Dict[str, Any]]:
    cursor.execute(query)
    df = as_pandas(cursor)
    df = clean_dataframe(df)
    records = []
    for row in df.itertuples(index=False):
        record = transform_fn(row)
        if record is not None:
            records.append(record)
    return records


# Extraction
print("Extracting data from Impala...")
customer_records = extract_and_transform(
    cursor, _query_customer(N_CLIENTI, N_MINIMO_CONNESSIONI, N_MESI), transform_customer
)
print(f"  Customer: {len(customer_records)} records")

salesrep_records = extract_and_transform(
    cursor, _query_salesrep(N_CLIENTI, N_MINIMO_CONNESSIONI, N_MESI), transform_salesrep
)
print(f"  SalesRep: {len(salesrep_records)} records")

product_records = extract_and_transform(
    cursor, _query_product(N_CLIENTI, N_MINIMO_CONNESSIONI, N_MESI), transform_product
)
print(f"  Product: {len(product_records)} records")

customer_salesrep_records = extract_and_transform(
    cursor,
    _query_customer_salesrep(N_CLIENTI, N_MINIMO_CONNESSIONI, N_MESI),
    transform_customer_salesrep,
)
print(f"  Customer-SalesRep: {len(customer_salesrep_records)} records")

customer_product_records = extract_and_transform(
    cursor,
    _query_customer_product(N_CLIENTI, N_MINIMO_CONNESSIONI, N_MESI),
    transform_customer_product,
)
print(f"  Customer-Product: {len(customer_product_records)} records")


# ============================================================
# NEO4J
# ============================================================


def chunked(data: List[Dict[str, Any]], size: int):
    for i in range(0, len(data), size):
        yield data[i : i + size]


# Constraints
constraints = [
    "CREATE CONSTRAINT customer_id IF NOT EXISTS FOR (n:Customer) REQUIRE n.ids_customer IS UNIQUE",
    "CREATE CONSTRAINT sector_id IF NOT EXISTS FOR (n:Sector) REQUIRE n.att_sector IS UNIQUE",
    "CREATE CONSTRAINT branch_id IF NOT EXISTS FOR (n:Branch) REQUIRE n.att_branch IS UNIQUE",
    "CREATE CONSTRAINT channel_id IF NOT EXISTS FOR (n:Channel) REQUIRE n.att_saleschannel IS UNIQUE",
    "CREATE CONSTRAINT marketsegment_id IF NOT EXISTS FOR (n:MarketSegment) REQUIRE n.att_marketsegment IS UNIQUE",
    "CREATE CONSTRAINT city_id IF NOT EXISTS FOR (n:City) REQUIRE n.att_city IS UNIQUE",
    "CREATE CONSTRAINT county_id IF NOT EXISTS FOR (n:County) REQUIRE n.att_county IS UNIQUE",
    "CREATE CONSTRAINT country_id IF NOT EXISTS FOR (n:Country) REQUIRE n.att_country IS UNIQUE",
    "CREATE CONSTRAINT salesrep_id IF NOT EXISTS FOR (n:SalesRep) REQUIRE n.ids_salesrep IS UNIQUE",
    "CREATE CONSTRAINT job_id IF NOT EXISTS FOR (n:Job) REQUIRE n.dsc_job IS UNIQUE",
    "CREATE CONSTRAINT district_id IF NOT EXISTS FOR (n:District) REQUIRE n.ids_district IS UNIQUE",
    "CREATE CONSTRAINT territory_id IF NOT EXISTS FOR (n:Territory) REQUIRE n.ids_territory IS UNIQUE",
    "CREATE CONSTRAINT macroregion_id IF NOT EXISTS FOR (n:MacroRegion) REQUIRE n.ids_mregion IS UNIQUE",
    "CREATE CONSTRAINT level1_id IF NOT EXISTS FOR (n:Level1) REQUIRE n.level1id IS UNIQUE",
    "CREATE CONSTRAINT level2_id IF NOT EXISTS FOR (n:Level2) REQUIRE n.level2id IS UNIQUE",
    "CREATE CONSTRAINT level3_id IF NOT EXISTS FOR (n:Level3) REQUIRE n.level3id IS UNIQUE",
    "CREATE CONSTRAINT level4_id IF NOT EXISTS FOR (n:Level4) REQUIRE n.level4id IS UNIQUE",
    "CREATE CONSTRAINT level5_id IF NOT EXISTS FOR (n:Level5) REQUIRE n.level5id IS UNIQUE",
]

with driver.session(database=database) as session:
    for q in tqdm(constraints, desc="Constraints"):
        session.run(q)


# ============================================================
# NODES
# ============================================================


# Customer
def load_customer(tx, rows):
    query = """
    UNWIND $rows AS row
    MERGE (c:Customer {ids_customer: row.customerId})
    SET c.dsc_customer = row.customer_name,
        c.partita_iva = row.customer_partitaIva,
        c.att_numberemployees = row.customer_numberEmployees,
        c.att_potential = row.customer_potential,
        c.att_creditlimit = row.customer_creditLimit,
        c.att_payterms = row.customer_payTerms,
        c.att_paytermsdescription = row.customer_payTermsDescription,
        c.att_paymentmethod = row.customer_paymentMethod,
        c.att_paymentmethoddescription = row.customer_paymentMethodDescription,
        c.att_email = row.customer_email,
        c.att_street = row.customer_street,
        c.att_zipcode = row.customer_zipcode,
        c.att_longitude_x = row.customer_longitude,
        c.att_latitude_y = row.customer_latitude
    """
    tx.run(query, rows=rows)


with driver.session(database=database) as session:
    for batch in tqdm(chunked(customer_records, size=1000), desc="Nodo Customer"):
        session.execute_write(load_customer, batch)


# Sector
def load_sector(tx, rows):
    query = """
    UNWIND $rows AS row
    WITH row WHERE row.sectorId IS NOT NULL
    MERGE (s:Sector {att_sector: row.sectorId})
    SET s.att_sectordescription = row.sectorDescription
    """
    tx.run(query, rows=rows)


with driver.session(database=database) as session:
    for batch in tqdm(chunked(customer_records, size=1000), desc="Nodo Sector"):
        session.execute_write(load_sector, batch)


# Branch
def load_branch(tx, rows):
    query = """
    UNWIND $rows AS row
    WITH row WHERE row.branchId IS NOT NULL
    MERGE (b:Branch {att_branch: row.branchId})
    SET b.att_branchdescription = row.branchDescription
    """
    tx.run(query, rows=rows)


with driver.session(database=database) as session:
    for batch in tqdm(chunked(customer_records, size=1000), desc="Nodo Branch"):
        session.execute_write(load_branch, batch)


# Channel
def load_channel(tx, rows):
    query = """
    UNWIND $rows AS row
    WITH row WHERE row.channelId IS NOT NULL
    MERGE (ch:Channel {att_saleschannel: row.channelId})
    SET ch.dsc_channel = row.channelDescription
    """
    tx.run(query, rows=rows)


with driver.session(database=database) as session:
    for batch in tqdm(chunked(customer_records, size=1000), desc="Nodo Channel"):
        session.execute_write(load_channel, batch)


# MarketSegment
def load_marketsegment(tx, rows):
    query = """
    UNWIND $rows AS row
    WITH row WHERE row.marketSegmentId IS NOT NULL
    MERGE (ms:MarketSegment {att_marketsegment: row.marketSegmentId})
    SET ms.att_marketsegmentdescription = row.marketSegmentDescription
    """
    tx.run(query, rows=rows)


with driver.session(database=database) as session:
    for batch in tqdm(chunked(customer_records, size=1000), desc="Nodo MarketSegment"):
        session.execute_write(load_marketsegment, batch)


# City
def load_city(tx, rows):
    query = """
    UNWIND $rows AS row
    WITH row WHERE row.cityName IS NOT NULL
    MERGE (:City {att_city: row.cityName})
    """
    tx.run(query, rows=rows)


with driver.session(database=database) as session:
    for batch in tqdm(chunked(customer_records, size=1000), desc="Nodo City"):
        session.execute_write(load_city, batch)


# County
def load_county(tx, rows):
    query = """
    UNWIND $rows AS row
    WITH row WHERE row.countyName IS NOT NULL
    MERGE (:County {att_county: row.countyName})
    """
    tx.run(query, rows=rows)


with driver.session(database=database) as session:
    for batch in tqdm(chunked(customer_records, size=1000), desc="Nodo County"):
        session.execute_write(load_county, batch)


# Country
def load_country(tx, rows):
    query = """
    UNWIND $rows AS row
    WITH row WHERE row.countryName IS NOT NULL
    MERGE (:Country {att_country: row.countryName})
    """
    tx.run(query, rows=rows)


with driver.session(database=database) as session:
    for batch in tqdm(chunked(customer_records, size=1000), desc="Nodo Country"):
        session.execute_write(load_country, batch)


# SalesRep
def load_salesrep(tx, rows):
    query = """
    UNWIND $rows AS row
    MERGE (sr:SalesRep {ids_salesrep: row.salesrepId})
    SET sr.dsc_salesrepfirstname = row.salesrep_firstname,
        sr.dsc_salesrepsurname = row.salesrep_surname,
        sr.att_mobilephonenumber = row.salesrep_mobile,
        sr.att_email = row.salesrep_email,
        sr.att_street = row.salesrep_street,
        sr.att_zipcode = row.salesrep_zipcode
    """
    tx.run(query, rows=rows)


with driver.session(database=database) as session:
    for batch in tqdm(chunked(salesrep_records, size=1000), desc="Nodo SalesRep"):
        session.execute_write(load_salesrep, batch)


# Job
def load_job(tx, rows):
    query = """
    UNWIND $rows AS row
    WITH row WHERE row.jobDescription IS NOT NULL
    MERGE (:Job {dsc_job: row.jobDescription})
    """
    tx.run(query, rows=rows)


with driver.session(database=database) as session:
    for batch in tqdm(chunked(salesrep_records, size=1000), desc="Nodo Job"):
        session.execute_write(load_job, batch)


# District
def load_district(tx, rows):
    query = """
    UNWIND $rows AS row
    WITH row WHERE row.districtId IS NOT NULL
    MERGE (d:District {ids_district: row.districtId})
    SET d.dsc_district = row.districtName
    """
    tx.run(query, rows=rows)


with driver.session(database=database) as session:
    for batch in tqdm(chunked(salesrep_records, size=1000), desc="Nodo District"):
        session.execute_write(load_district, batch)


# Territory
def load_territory(tx, rows):
    query = """
    UNWIND $rows AS row
    WITH row WHERE row.territoryId IS NOT NULL
    MERGE (t:Territory {ids_territory: row.territoryId})
    SET t.dsc_territory = row.territoryName
    """
    tx.run(query, rows=rows)


with driver.session(database=database) as session:
    for batch in tqdm(chunked(salesrep_records, size=1000), desc="Nodo Territory"):
        session.execute_write(load_territory, batch)


# MacroRegion
def load_macroregion(tx, rows):
    query = """
    UNWIND $rows AS row
    WITH row WHERE row.mregionId IS NOT NULL
    MERGE (mr:MacroRegion {ids_mregion: row.mregionId})
    SET mr.dsc_mregion = row.mregionName
    """
    tx.run(query, rows=rows)


with driver.session(database=database) as session:
    for batch in tqdm(chunked(salesrep_records, size=1000), desc="Nodo MacroRegion"):
        session.execute_write(load_macroregion, batch)


# Product hierarchy (Level1-5)
def load_product_hierarchy(tx, rows):
    query = """
    UNWIND $rows AS row

    FOREACH (_ IN CASE WHEN row.level1Id IS NOT NULL THEN [1] ELSE [] END |
        MERGE (l1:Level1 {level1id: row.level1Id})
        SET l1.level1 = row.level1_name
    )
    FOREACH (_ IN CASE WHEN row.level2Id IS NOT NULL THEN [1] ELSE [] END |
        MERGE (l2:Level2 {level2id: row.level2Id})
        SET l2.level2 = row.level2_name
    )
    FOREACH (_ IN CASE WHEN row.level3Id IS NOT NULL THEN [1] ELSE [] END |
        MERGE (l3:Level3 {level3id: row.level3Id})
        SET l3.level3 = row.level3_name
    )
    FOREACH (_ IN CASE WHEN row.level4Id IS NOT NULL THEN [1] ELSE [] END |
        MERGE (l4:Level4 {level4id: row.level4Id})
        SET l4.level4 = row.level4_name
    )

    MERGE (l5:Level5 {level5id: row.level5Id})
    SET l5.level5 = row.level5_name
    """
    tx.run(query, rows=rows)


with driver.session(database=database) as session:
    for batch in tqdm(chunked(product_records, size=1000), desc="Nodo Product"):
        session.execute_write(load_product_hierarchy, batch)


# ============================================================
# RELATIONSHIPS
# ============================================================


# Customer -[:IN_SECTOR]-> Sector
def rel_customer_sector(tx, rows):
    query = """
    UNWIND $rows AS row
    WITH row WHERE row.sectorId IS NOT NULL
    MATCH (c:Customer {ids_customer: row.customerId})
    MATCH (s:Sector {att_sector: row.sectorId})
    MERGE (c)-[:IN_SECTOR]->(s)
    """
    tx.run(query, rows=rows)


with driver.session(database=database) as session:
    for batch in tqdm(chunked(customer_records, size=1000), desc="Rel Customer-IN_SECTOR->Sector"):
        session.execute_write(rel_customer_sector, batch)


# Customer -[:IN_BRANCH]-> Branch
def rel_customer_branch(tx, rows):
    query = """
    UNWIND $rows AS row
    WITH row WHERE row.branchId IS NOT NULL
    MATCH (c:Customer {ids_customer: row.customerId})
    MATCH (b:Branch {att_branch: row.branchId})
    MERGE (c)-[:IN_BRANCH]->(b)
    """
    tx.run(query, rows=rows)


with driver.session(database=database) as session:
    for batch in tqdm(chunked(customer_records, size=1000), desc="Rel Customer-IN_BRANCH->Branch"):
        session.execute_write(rel_customer_branch, batch)


# Customer -[:IN_CHANNEL]-> Channel
def rel_customer_channel(tx, rows):
    query = """
    UNWIND $rows AS row
    WITH row WHERE row.channelId IS NOT NULL
    MATCH (c:Customer {ids_customer: row.customerId})
    MATCH (ch:Channel {att_saleschannel: row.channelId})
    MERGE (c)-[:IN_CHANNEL]->(ch)
    """
    tx.run(query, rows=rows)


with driver.session(database=database) as session:
    for batch in tqdm(
        chunked(customer_records, size=1000), desc="Rel Customer-IN_CHANNEL->Channel"
    ):
        session.execute_write(rel_customer_channel, batch)


# Customer -[:IN_MARKET_SEGMENT]-> MarketSegment
def rel_customer_marketsegment(tx, rows):
    query = """
    UNWIND $rows AS row
    WITH row WHERE row.marketSegmentId IS NOT NULL
    MATCH (c:Customer {ids_customer: row.customerId})
    MATCH (ms:MarketSegment {att_marketsegment: row.marketSegmentId})
    MERGE (c)-[:IN_MARKET_SEGMENT]->(ms)
    """
    tx.run(query, rows=rows)


with driver.session(database=database) as session:
    for batch in tqdm(
        chunked(customer_records, size=1000), desc="Rel Customer-IN_MARKET_SEGMENT->MarketSegment"
    ):
        session.execute_write(rel_customer_marketsegment, batch)


# Customer -[:LOCATED_IN]-> City
def rel_customer_city(tx, rows):
    query = """
    UNWIND $rows AS row
    WITH row WHERE row.cityName IS NOT NULL
    MATCH (c:Customer {ids_customer: row.customerId})
    MATCH (city:City {att_city: row.cityName})
    MERGE (c)-[:LOCATED_IN]->(city)
    """
    tx.run(query, rows=rows)


with driver.session(database=database) as session:
    for batch in tqdm(chunked(customer_records, size=1000), desc="Rel Customer-LOCATED_IN->City"):
        session.execute_write(rel_customer_city, batch)


# City -[:IN_COUNTY]-> County
def rel_city_county(tx, rows):
    query = """
    UNWIND $rows AS row
    WITH row WHERE row.cityName IS NOT NULL AND row.countyName IS NOT NULL
    MATCH (city:City {att_city: row.cityName})
    MATCH (county:County {att_county: row.countyName})
    MERGE (city)-[:IN_COUNTY]->(county)
    """
    tx.run(query, rows=rows)


with driver.session(database=database) as session:
    for batch in tqdm(chunked(customer_records, size=1000), desc="Rel City-IN_COUNTY->County"):
        session.execute_write(rel_city_county, batch)


# County -[:IN_COUNTRY]-> Country
def rel_county_country(tx, rows):
    query = """
    UNWIND $rows AS row
    WITH row WHERE row.countyName IS NOT NULL AND row.countryName IS NOT NULL
    MATCH (county:County {att_county: row.countyName})
    MATCH (country:Country {att_country: row.countryName})
    MERGE (county)-[:IN_COUNTRY]->(country)
    """
    tx.run(query, rows=rows)


with driver.session(database=database) as session:
    for batch in tqdm(chunked(customer_records, size=1000), desc="Rel County-IN_COUNTRY->Country"):
        session.execute_write(rel_county_country, batch)


# SalesRep -[:TYPE]-> Job
def rel_salesrep_job(tx, rows):
    query = """
    UNWIND $rows AS row
    WITH row WHERE row.jobDescription IS NOT NULL
    MATCH (sr:SalesRep {ids_salesrep: row.salesrepId})
    MATCH (j:Job {dsc_job: row.jobDescription})
    MERGE (sr)-[:TYPE]->(j)
    """
    tx.run(query, rows=rows)


with driver.session(database=database) as session:
    for batch in tqdm(chunked(salesrep_records, size=1000), desc="Rel SalesRep-TYPE->Job"):
        session.execute_write(rel_salesrep_job, batch)


# SalesRep -[:IN_CITY]-> City
def rel_salesrep_city(tx, rows):
    query = """
    UNWIND $rows AS row
    WITH row WHERE row.salesrep_city IS NOT NULL
    MATCH (sr:SalesRep {ids_salesrep: row.salesrepId})
    MATCH (city:City {att_city: row.salesrep_city})
    MERGE (sr)-[:IN_CITY]->(city)
    """
    tx.run(query, rows=rows)


with driver.session(database=database) as session:
    for batch in tqdm(chunked(salesrep_records, size=1000), desc="Rel SalesRep-IN_CITY->City"):
        session.execute_write(rel_salesrep_city, batch)


# SalesRep -[:IN_CHANNEL]-> Channel
def rel_salesrep_channel(tx, rows):
    query = """
    UNWIND $rows AS row
    WITH row WHERE row.channelDescription IS NOT NULL
    MATCH (sr:SalesRep {ids_salesrep: row.salesrepId})
    MATCH (ch:Channel {dsc_channel: row.channelDescription})
    MERGE (sr)-[:IN_CHANNEL]->(ch)
    """
    tx.run(query, rows=rows)


with driver.session(database=database) as session:
    for batch in tqdm(
        chunked(salesrep_records, size=1000), desc="Rel SalesRep-IN_CHANNEL->Channel"
    ):
        session.execute_write(rel_salesrep_channel, batch)


# SalesRep -[:IN_SECTOR]-> Sector
def rel_salesrep_sector(tx, rows):
    query = """
    UNWIND $rows AS row
    WITH row WHERE row.sectorDescription IS NOT NULL
    MATCH (sr:SalesRep {ids_salesrep: row.salesrepId})
    MATCH (s:Sector {att_sector: row.sectorDescription})
    MERGE (sr)-[:IN_SECTOR]->(s)
    """
    tx.run(query, rows=rows)


with driver.session(database=database) as session:
    for batch in tqdm(chunked(salesrep_records, size=1000), desc="Rel SalesRep-IN_SECTOR->Sector"):
        session.execute_write(rel_salesrep_sector, batch)


# SalesRep -[:IN_MARKET_SEGMENT]-> MarketSegment
def rel_salesrep_marketsegment(tx, rows):
    query = """
    UNWIND $rows AS row
    WITH row WHERE row.marketSegmentId IS NOT NULL
    MATCH (sr:SalesRep {ids_salesrep: row.salesrepId})
    MATCH (ms:MarketSegment {att_marketsegment: row.marketSegmentId})
    MERGE (sr)-[:IN_MARKET_SEGMENT]->(ms)
    """
    tx.run(query, rows=rows)


with driver.session(database=database) as session:
    for batch in tqdm(
        chunked(salesrep_records, size=1000), desc="Rel SalesRep-IN_MARKET_SEGMENT->MarketSegment"
    ):
        session.execute_write(rel_salesrep_marketsegment, batch)


# SalesRep -[:IN_DISTRICT]-> District
def rel_salesrep_district(tx, rows):
    query = """
    UNWIND $rows AS row
    WITH row WHERE row.districtId IS NOT NULL
    MATCH (sr:SalesRep {ids_salesrep: row.salesrepId})
    MATCH (d:District {ids_district: row.districtId})
    MERGE (sr)-[:IN_DISTRICT]->(d)
    """
    tx.run(query, rows=rows)


with driver.session(database=database) as session:
    for batch in tqdm(
        chunked(salesrep_records, size=1000), desc="Rel SalesRep-IN_DISTRICT->District"
    ):
        session.execute_write(rel_salesrep_district, batch)


# SalesRep -[:HAS_CUSTOMER]-> Customer
def rel_salesrep_customer(tx, rows):
    query = """
    UNWIND $rows AS row
    MATCH (sr:SalesRep {ids_salesrep: row.salesrepId})
    MATCH (c:Customer {ids_customer: row.customerId})
    MERGE (sr)-[:HAS_CUSTOMER]->(c)
    """
    tx.run(query, rows=rows)


with driver.session(database=database) as session:
    for batch in tqdm(
        chunked(customer_salesrep_records, size=1000), desc="Rel SalesRep-HAS_CUSTOMER->Customer"
    ):
        session.execute_write(rel_salesrep_customer, batch)


# District -[:IN_TERRITORY]-> Territory
def rel_district_territory(tx, rows):
    query = """
    UNWIND $rows AS row
    WITH row WHERE row.districtId IS NOT NULL AND row.territoryId IS NOT NULL
    MATCH (d:District {ids_district: row.districtId})
    MATCH (t:Territory {ids_territory: row.territoryId})
    MERGE (d)-[:IN_TERRITORY]->(t)
    """
    tx.run(query, rows=rows)


with driver.session(database=database) as session:
    for batch in tqdm(
        chunked(salesrep_records, size=1000), desc="Rel District-IN_TERRITORY->Territory"
    ):
        session.execute_write(rel_district_territory, batch)


# Territory -[:IN_MACRO_REGION]-> MacroRegion
def rel_territory_macroregion(tx, rows):
    query = """
    UNWIND $rows AS row
    WITH row WHERE row.territoryId IS NOT NULL AND row.mregionId IS NOT NULL
    MATCH (t:Territory {ids_territory: row.territoryId})
    MATCH (mr:MacroRegion {ids_mregion: row.mregionId})
    MERGE (t)-[:IN_MACRO_REGION]->(mr)
    """
    tx.run(query, rows=rows)


with driver.session(database=database) as session:
    for batch in tqdm(
        chunked(salesrep_records, size=1000), desc="Rel Territory-IN_MACRO_REGION->MacroRegion"
    ):
        session.execute_write(rel_territory_macroregion, batch)


# District -[:MANAGED_BY]-> SalesRep
def rel_district_manager(tx, rows):
    query = """
    UNWIND $rows AS row
    WITH row WHERE row.districtId IS NOT NULL AND row.districtSalesrepId IS NOT NULL
    MATCH (d:District {ids_district: row.districtId})
    MERGE (mgr:SalesRep {ids_salesrep: row.districtSalesrepId})
    ON CREATE SET mgr.dsc_salesrepfirstname = row.districtSalesrepFirstname,
                  mgr.dsc_salesrepsurname = row.districtSalesrepSurname
    MERGE (d)-[:MANAGED_BY]->(mgr)
    """
    tx.run(query, rows=rows)


with driver.session(database=database) as session:
    for batch in tqdm(
        chunked(salesrep_records, size=1000), desc="Rel District-MANAGED_BY->SalesRep"
    ):
        session.execute_write(rel_district_manager, batch)


# Territory -[:MANAGED_BY]-> SalesRep
def rel_territory_manager(tx, rows):
    query = """
    UNWIND $rows AS row
    WITH row WHERE row.territoryId IS NOT NULL AND row.territorySalesrepId IS NOT NULL
    MATCH (t:Territory {ids_territory: row.territoryId})
    MERGE (mgr:SalesRep {ids_salesrep: row.territorySalesrepId})
    ON CREATE SET mgr.dsc_salesrepfirstname = row.territorySalesrepFirstname,
                  mgr.dsc_salesrepsurname = row.territorySalesrepSurname
    MERGE (t)-[:MANAGED_BY]->(mgr)
    """
    tx.run(query, rows=rows)


with driver.session(database=database) as session:
    for batch in tqdm(
        chunked(salesrep_records, size=1000), desc="Rel Territory-MANAGED_BY->SalesRep"
    ):
        session.execute_write(rel_territory_manager, batch)


# MacroRegion -[:MANAGED_BY]-> SalesRep
def rel_macroregion_manager(tx, rows):
    query = """
    UNWIND $rows AS row
    WITH row WHERE row.mregionId IS NOT NULL AND row.mregionSalesrepId IS NOT NULL
    MATCH (mr:MacroRegion {ids_mregion: row.mregionId})
    MERGE (mgr:SalesRep {ids_salesrep: row.mregionSalesrepId})
    ON CREATE SET mgr.dsc_salesrepfirstname = row.mregionSalesrepFirstname,
                  mgr.dsc_salesrepsurname = row.mregionSalesrepSurname
    MERGE (mr)-[:MANAGED_BY]->(mgr)
    """
    tx.run(query, rows=rows)


with driver.session(database=database) as session:
    for batch in tqdm(
        chunked(salesrep_records, size=1000), desc="Rel MacroRegion-MANAGED_BY->SalesRep"
    ):
        session.execute_write(rel_macroregion_manager, batch)


# Level5 -[:BELONGS_TO]-> Level4 -> Level3 -> Level2 -> Level1
def rel_l5_l4(tx, rows):
    query = """
    UNWIND $rows AS row
    WITH row WHERE row.level5Id IS NOT NULL AND row.level4Id IS NOT NULL
    MATCH (l5:Level5 {level5id: row.level5Id})
    MATCH (l4:Level4 {level4id: row.level4Id})
    MERGE (l5)-[:BELONGS_TO]->(l4)
    """
    tx.run(query, rows=rows)


def rel_l4_l3(tx, rows):
    query = """
    UNWIND $rows AS row
    WITH row WHERE row.level4Id IS NOT NULL AND row.level3Id IS NOT NULL
    MATCH (l4:Level4 {level4id: row.level4Id})
    MATCH (l3:Level3 {level3id: row.level3Id})
    MERGE (l4)-[:BELONGS_TO]->(l3)
    """
    tx.run(query, rows=rows)


def rel_l3_l2(tx, rows):
    query = """
    UNWIND $rows AS row
    WITH row WHERE row.level3Id IS NOT NULL AND row.level2Id IS NOT NULL
    MATCH (l3:Level3 {level3id: row.level3Id})
    MATCH (l2:Level2 {level2id: row.level2Id})
    MERGE (l3)-[:BELONGS_TO]->(l2)
    """
    tx.run(query, rows=rows)


def rel_l2_l1(tx, rows):
    query = """
    UNWIND $rows AS row
    WITH row WHERE row.level2Id IS NOT NULL AND row.level1Id IS NOT NULL
    MATCH (l2:Level2 {level2id: row.level2Id})
    MATCH (l1:Level1 {level1id: row.level1Id})
    MERGE (l2)-[:BELONGS_TO]->(l1)
    """
    tx.run(query, rows=rows)


with driver.session(database=database) as session:
    for batch in tqdm(chunked(product_records, size=1000), desc="Rel Product BELONGS_TO"):
        session.execute_write(rel_l5_l4, batch)
        session.execute_write(rel_l4_l3, batch)
        session.execute_write(rel_l3_l2, batch)
        session.execute_write(rel_l2_l1, batch)


# Customer -[:BUYS]-> Level5
def rel_customer_buys(tx, rows):
    query = """
    UNWIND $rows AS row
    MATCH (c:Customer {ids_customer: row.customerId})
    MATCH (l5:Level5 {level5id: row.level5Id})
    MERGE (c)-[r:BUYS]->(l5)
    ON CREATE SET r.first_purchase = row.dat
    ON MATCH SET r.last_purchase = row.dat
    """
    tx.run(query, rows=rows)


with driver.session(database=database) as session:
    for batch in tqdm(
        chunked(customer_product_records, size=1000), desc="Rel Customer-BUYS->Level5"
    ):
        session.execute_write(rel_customer_buys, batch)


# Channel -[:IN_SECTOR]-> Sector
def rel_channel_sector(tx, rows):
    query = """
    UNWIND $rows AS row
    WITH row
    WHERE row.channelId IS NOT NULL AND row.sectorId IS NOT NULL
    MATCH (ch:Channel {att_saleschannel: row.channelId})
    MATCH (s:Sector {att_sector: row.sectorId})
    MERGE (ch)-[:IN_SECTOR]->(s)
    """
    tx.run(query, rows=rows)


with driver.session(database=database) as session:
    for batch in tqdm(
        chunked(customer_records, size=1000),
        desc="Rel Channel-IN_SECTOR->Sector"
    ):
        session.execute_write(rel_channel_sector, batch)

# Sector -[:IN_MARKET_SEGMENT]-> MarketSegment
def rel_sector_marketsegment(tx, rows):
    query = """
    UNWIND $rows AS row
    WITH row
    WHERE row.sectorId IS NOT NULL AND row.marketSegmentId IS NOT NULL
    MATCH (s:Sector {att_sector: row.sectorId})
    MATCH (ms:MarketSegment {att_marketsegment: row.marketSegmentId})
    MERGE (s)-[:IN_MARKET_SEGMENT]->(ms)
    """
    tx.run(query, rows=rows)


with driver.session(database=database) as session:
    for batch in tqdm(
        chunked(customer_records, size=1000),
        desc="Rel Sector-IN_MARKET_SEGMENT->MarketSegment"
    ):
        session.execute_write(rel_sector_marketsegment, batch)



# ============================================================
# CHIUSURA
# ============================================================
cursor.close()
connection.close()
driver.close()
print("Done! Grafo caricato con successo.")
