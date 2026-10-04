"""Load the local synthetic CSV tables into the Neo4j graph schema.

Unlike ``ETL_fact.py``, this entry point does not require the unfinished
Impala extraction layer. It is intentionally idempotent and can be run again
against the same Neo4j database.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from config import config

CONSTRAINTS = [
    "CREATE CONSTRAINT customer_id IF NOT EXISTS FOR (n:Customer) REQUIRE n.ids_customer IS UNIQUE",
    "CREATE CONSTRAINT sector_id IF NOT EXISTS FOR (n:Sector) REQUIRE n.att_sector IS UNIQUE",
    "CREATE CONSTRAINT branch_id IF NOT EXISTS FOR (n:Branch) REQUIRE n.att_branch IS UNIQUE",
    "CREATE CONSTRAINT channel_id IF NOT EXISTS FOR (n:Channel) REQUIRE n.att_saleschannel IS UNIQUE",
    (
        "CREATE CONSTRAINT marketsegment_id IF NOT EXISTS "
        "FOR (n:MarketSegment) REQUIRE n.att_marketsegment IS UNIQUE"
    ),
    "CREATE CONSTRAINT city_id IF NOT EXISTS FOR (n:City) REQUIRE n.att_city IS UNIQUE",
    "CREATE CONSTRAINT county_id IF NOT EXISTS FOR (n:County) REQUIRE n.att_county IS UNIQUE",
    "CREATE CONSTRAINT country_id IF NOT EXISTS FOR (n:Country) REQUIRE n.att_country IS UNIQUE",
    "CREATE CONSTRAINT salesrep_id IF NOT EXISTS FOR (n:SalesRep) REQUIRE n.ids_salesrep IS UNIQUE",
    "CREATE CONSTRAINT district_id IF NOT EXISTS FOR (n:District) REQUIRE n.ids_district IS UNIQUE",
    "CREATE CONSTRAINT territory_id IF NOT EXISTS FOR (n:Territory) REQUIRE n.ids_territory IS UNIQUE",
    "CREATE CONSTRAINT macroregion_id IF NOT EXISTS FOR (n:MacroRegion) REQUIRE n.ids_mregion IS UNIQUE",
    "CREATE CONSTRAINT level1_id IF NOT EXISTS FOR (n:Level1) REQUIRE n.level1id IS UNIQUE",
    "CREATE CONSTRAINT level2_id IF NOT EXISTS FOR (n:Level2) REQUIRE n.level2id IS UNIQUE",
    "CREATE CONSTRAINT level3_id IF NOT EXISTS FOR (n:Level3) REQUIRE n.level3id IS UNIQUE",
    "CREATE CONSTRAINT level4_id IF NOT EXISTS FOR (n:Level4) REQUIRE n.level4id IS UNIQUE",
    "CREATE CONSTRAINT level5_id IF NOT EXISTS FOR (n:Level5) REQUIRE n.level5id IS UNIQUE",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"File mancante: {path}. Esegui prima make synthetic-data.")
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def batches(rows: list[dict[str, Any]], size: int = 1_000) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def execute_batches(session: Any, query: str, rows: list[dict[str, Any]]) -> None:
    for batch in batches(rows):
        session.execute_write(lambda tx, current_batch=batch: tx.run(query, rows=current_batch).consume())


def aggregate_purchases(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Collapse events only for the graph edge while retaining event counts."""
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row["ids_customer"], row["level5id"])
        current = grouped.setdefault(
            key,
            {
                "customerId": key[0],
                "level5Id": key[1],
                "first_purchase": row["dat"],
                "last_purchase": row["dat"],
                "purchase_count": 0,
                "total_quantity": 0,
            },
        )
        current["first_purchase"] = min(current["first_purchase"], row["dat"])
        current["last_purchase"] = max(current["last_purchase"], row["dat"])
        current["purchase_count"] += 1
        current["total_quantity"] += int(row["quantity"])
    return list(grouped.values())


CUSTOMER_QUERY = """
UNWIND $rows AS row
MERGE (c:Customer {ids_customer: row.ids_customer})
SET c.dsc_customer = row.dsc_customer,
    c.partita_iva = row.partita_iva,
    c.att_numberemployees = toInteger(row.att_numberemployees),
    c.att_potential = row.att_potential,
    c.att_creditlimit = toFloat(row.att_creditlimit),
    c.att_payterms = row.att_payterms,
    c.att_paytermsdescription = row.att_paytermsdescription,
    c.att_paymentmethod = row.att_paymentmethod,
    c.att_paymentmethoddescription = row.att_paymentmethoddescription,
    c.att_email = row.att_email,
    c.att_street = row.att_street,
    c.att_zipcode = row.att_zipcode,
    c.att_longitude_x = toFloat(row.att_longitude_x),
    c.att_latitude_y = toFloat(row.att_latitude_y)
MERGE (s:Sector {att_sector: row.att_sector})
SET s.att_sectordescription = row.att_sectordescription
MERGE (b:Branch {att_branch: row.att_branch})
SET b.att_branchdescription = row.att_branchdescription
MERGE (ch:Channel {att_saleschannel: row.att_saleschannel})
SET ch.dsc_channel = row.dsc_channel
MERGE (ms:MarketSegment {att_marketsegment: row.att_marketsegment})
SET ms.att_marketsegmentdescription = row.att_marketsegmentdescription
MERGE (city:City {att_city: row.att_city})
MERGE (county:County {att_county: row.att_county})
MERGE (country:Country {att_country: row.att_country})
MERGE (c)-[:IN_SECTOR]->(s)
MERGE (c)-[:IN_BRANCH]->(b)
MERGE (c)-[:IN_CHANNEL]->(ch)
MERGE (c)-[:IN_MARKET_SEGMENT]->(ms)
MERGE (c)-[:LOCATED_IN]->(city)
MERGE (city)-[:IN_COUNTY]->(county)
MERGE (county)-[:IN_COUNTRY]->(country)
MERGE (ch)-[:IN_SECTOR]->(s)
MERGE (s)-[:IN_MARKET_SEGMENT]->(ms)
"""

SALESREP_QUERY = """
UNWIND $rows AS row
MERGE (sr:SalesRep {ids_salesrep: row.ids_salesrep})
SET sr.dsc_salesrepfirstname = row.dsc_salesrepfirstname,
    sr.dsc_salesrepsurname = row.dsc_salesrepsurname,
    sr.att_mobilephonenumber = row.att_mobilephonenumber,
    sr.att_email = row.att_email,
    sr.att_street = row.att_street,
    sr.att_zipcode = row.att_zipcode
MERGE (job:Job {dsc_job: row.dsc_job})
MERGE (city:City {att_city: row.att_cityname})
MERGE (channel:Channel {att_saleschannel: row.att_saleschannel})
SET channel.dsc_channel = row.dsc_channel
MERGE (sector:Sector {att_sector: row.att_sector})
SET sector.att_sectordescription = row.dsc_sector
MERGE (segment:MarketSegment {att_marketsegment: row.att_marketsegment})
MERGE (district:District {ids_district: row.ids_district})
SET district.dsc_district = row.dsc_district
MERGE (territory:Territory {ids_territory: row.ids_territory})
SET territory.dsc_territory = row.dsc_territory
MERGE (macro:MacroRegion {ids_mregion: row.ids_mregion})
SET macro.dsc_mregion = row.dsc_mregion
MERGE (district_manager:SalesRep {ids_salesrep: row.ids_districtsalesrep})
ON CREATE SET district_manager.dsc_salesrepfirstname = row.dsc_districtsalesrepfirstname,
              district_manager.dsc_salesrepsurname = row.dsc_districtsalesrepsurname
MERGE (territory_manager:SalesRep {ids_salesrep: row.ids_territorysalesrep})
ON CREATE SET territory_manager.dsc_salesrepfirstname = row.dsc_territorysalesrepfirstname,
              territory_manager.dsc_salesrepsurname = row.dsc_territorysalesrepsurname
MERGE (macro_manager:SalesRep {ids_salesrep: row.ids_mregionsalesrep})
ON CREATE SET macro_manager.dsc_salesrepfirstname = row.dsc_mregionsalesrepfirstname,
              macro_manager.dsc_salesrepsurname = row.dsc_mregionsalesrepsurname
MERGE (sr)-[:TYPE]->(job)
MERGE (sr)-[:IN_CITY]->(city)
MERGE (sr)-[:IN_CHANNEL]->(channel)
MERGE (sr)-[:IN_SECTOR]->(sector)
MERGE (sr)-[:IN_MARKET_SEGMENT]->(segment)
MERGE (sr)-[:IN_DISTRICT]->(district)
MERGE (district)-[:IN_TERRITORY]->(territory)
MERGE (territory)-[:IN_MACRO_REGION]->(macro)
MERGE (district)-[:MANAGED_BY]->(district_manager)
MERGE (territory)-[:MANAGED_BY]->(territory_manager)
MERGE (macro)-[:MANAGED_BY]->(macro_manager)
"""

PRODUCT_QUERY = """
UNWIND $rows AS row
MERGE (l1:Level1 {level1id: row.level1id}) SET l1.level1 = row.level1
MERGE (l2:Level2 {level2id: row.level2id}) SET l2.level2 = row.level2
MERGE (l3:Level3 {level3id: row.level3id}) SET l3.level3 = row.level3
MERGE (l4:Level4 {level4id: row.level4id}) SET l4.level4 = row.level4
MERGE (l5:Level5 {level5id: row.level5id}) SET l5.level5 = row.level5
MERGE (l5)-[:BELONGS_TO]->(l4)
MERGE (l4)-[:BELONGS_TO]->(l3)
MERGE (l3)-[:BELONGS_TO]->(l2)
MERGE (l2)-[:BELONGS_TO]->(l1)
"""

CUSTOMER_SALESREP_QUERY = """
UNWIND $rows AS row
MATCH (c:Customer {ids_customer: row.ids_customer})
MATCH (sr:SalesRep {ids_salesrep: row.ids_salesrep})
MERGE (sr)-[:HAS_CUSTOMER]->(c)
"""

PURCHASE_QUERY = """
UNWIND $rows AS row
MATCH (c:Customer {ids_customer: row.customerId})
MATCH (l5:Level5 {level5id: row.level5Id})
MERGE (c)-[r:BUYS]->(l5)
SET r.first_purchase = date(row.first_purchase),
    r.last_purchase = date(row.last_purchase),
    r.purchase_count = row.purchase_count,
    r.total_quantity = row.total_quantity
"""

RECOMMENDATION_QUERY = """
UNWIND $rows AS row
MATCH (c:Customer {ids_customer: row.ids_customer})
MATCH (l5:Level5 {level5id: row.ids_product})
MERGE (l5)-[r:RECOMMENDED_TO]->(c)
SET r.product_name = row.product_name,
    r.rank = toInteger(row.rank),
    r.score = toFloat(row.score),
    r.source = row.source
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Load synthetic CSV tables into Neo4j.")
    parser.add_argument("--data-dir", type=Path, default=Path("database"))
    parser.add_argument(
        "--database",
        default=None,
        help="Neo4j database name (default: NEO4J_DATABASE or neo4j).",
    )
    parser.add_argument(
        "--include-placeholder-recommendations",
        action="store_true",
        help="Also create RECOMMENDED_TO edges from the placeholder table.",
    )
    args = parser.parse_args()
    database_name = args.database or os.getenv("NEO4J_DATABASE", "neo4j")

    customers = read_csv(args.data_dir / "customers.csv")
    salesreps = read_csv(args.data_dir / "salesreps.csv")
    products = read_csv(args.data_dir / "products.csv")
    assignments = read_csv(args.data_dir / "customer_salesreps.csv")
    purchases = read_csv(args.data_dir / "purchases.csv")
    recommendations = read_csv(args.data_dir / "recommendations_placeholder.csv")

    driver = config.neo4j_driver
    try:
        with driver.session(database=database_name) as session:
            for constraint in CONSTRAINTS:
                session.run(constraint).consume()
            execute_batches(session, CUSTOMER_QUERY, customers)
            execute_batches(session, SALESREP_QUERY, salesreps)
            execute_batches(session, PRODUCT_QUERY, products)
            execute_batches(session, CUSTOMER_SALESREP_QUERY, assignments)
            execute_batches(session, PURCHASE_QUERY, aggregate_purchases(purchases))
            if args.include_placeholder_recommendations:
                execute_batches(session, RECOMMENDATION_QUERY, recommendations)

            stats = session.run(
                """
                CALL () {
                    MATCH (c:Customer)
                    WHERE c.ids_customer STARTS WITH 'C-'
                    RETURN count(c) AS customers
                }
                CALL () {
                    MATCH (sr:SalesRep)
                    WHERE sr.ids_salesrep STARTS WITH 'SR-'
                    RETURN count(sr) AS salesreps
                }
                CALL () {
                    MATCH (p:Level5)
                    WHERE p.level5id STARTS WITH 'PRD-'
                    RETURN count(p) AS products
                }
                CALL () {
                    MATCH (c:Customer)-[b:BUYS]->(p:Level5)
                    WHERE c.ids_customer STARTS WITH 'C-' AND p.level5id STARTS WITH 'PRD-'
                    RETURN count(b) AS purchase_edges, coalesce(sum(b.purchase_count), 0) AS purchase_events
                }
                RETURN customers, salesreps, products, purchase_edges, purchase_events
                """
            ).single()
        print("Caricamento Neo4j completato")
        print(
            f"Customer: {stats['customers']} | SalesRep: {stats['salesreps']} | "
            f"Level5: {stats['products']} | BUYS: {stats['purchase_edges']} "
            f"({stats['purchase_events']} eventi)"
        )
        if args.include_placeholder_recommendations:
            print(f"Importate anche {len(recommendations)} raccomandazioni placeholder.")
        else:
            print("Raccomandazioni placeholder non importate: riservate alla futura GNN.")
    finally:
        config._shutdown()


if __name__ == "__main__":
    main()
