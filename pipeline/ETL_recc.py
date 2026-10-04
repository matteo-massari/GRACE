from dotenv import load_dotenv
from tqdm import tqdm
import csv
import os
from config import config

load_dotenv()

BATCH_SIZE = 1000
CSV_PATH = os.getenv("RECOMMENDATIONS_CSV_PATH", "database/gnn/rec_tables/10/dataset_res_10_features_True.csv")
DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
MAX_DEBUG_SAMPLES = 100


# =========================
# Utils
# =========================
def is_valid(value):
    if value is None:
        return False
    value_str = str(value).strip()
    return value_str not in ("", "NULL", "nan", "NaN", "None")


def normalize(value):
    if not is_valid(value):
        return None

    value_str = str(value).strip()

    if value_str.endswith(".0"):
        base = value_str[:-2]
        if base.isdigit():
            value_str = base

    if value_str.isdigit():
        return str(int(value_str))

    return value_str


def safe_value(value):
    return None if not is_valid(value) else str(value).strip()


def chunked(iterable, size):
    batch = []
    for item in iterable:
        if item is not None:
            batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def csv_row_to_recommendation_record(row):
    customer_id = normalize(row.get("ids_customer"))
    product_id = normalize(row.get("ids_product"))
    product_name = safe_value(row.get("product_name"))

    if customer_id is None or product_id is None:
        return None

    return {
        "customerId": customer_id,
        "level5Id": product_id,
        "productName": product_name,
    }


# =========================
# Cypher
# =========================
RECOMMENDATION_BATCH_QUERY = """
UNWIND $rows AS row

MATCH (c:Customer {ids_customer: row.customerId})
MATCH (l5:Level5 {level5id: row.level5Id})
MERGE (l5)-[r:RECOMMENDED_TO]->(c)
SET r.product_name = row.productName

RETURN count(r) AS created
"""

RECOMMENDATION_BATCH_QUERY_SAFE = """
UNWIND $rows AS row

OPTIONAL MATCH (c:Customer {ids_customer: row.customerId})
OPTIONAL MATCH (l5:Level5 {level5id: row.level5Id})

WITH row, c, l5,
     CASE WHEN c IS NOT NULL THEN 1 ELSE 0 END AS customerFound,
     CASE WHEN l5 IS NOT NULL THEN 1 ELSE 0 END AS level5Found

FOREACH (_ IN CASE WHEN c IS NOT NULL AND l5 IS NOT NULL THEN [1] ELSE [] END |
    MERGE (l5)-[r:RECOMMENDED_TO]->(c)
    SET r.product_name = row.productName
)

RETURN
    sum(customerFound) AS customer_found_rows,
    sum(level5Found) AS level5_found_rows,
    sum(CASE WHEN c IS NOT NULL AND l5 IS NOT NULL THEN 1 ELSE 0 END) AS matched_rows
"""


def write_recommendation_batch(tx, rows):
    result = tx.run(RECOMMENDATION_BATCH_QUERY_SAFE, rows=rows)
    record = result.single()

    if record is None:
        return {"customer_found_rows": 0, "level5_found_rows": 0, "matched_rows": 0}

    return {
        "customer_found_rows": record["customer_found_rows"],
        "level5_found_rows": record["level5_found_rows"],
        "matched_rows": record["matched_rows"],
    }


# =========================
# Debug
# =========================
def debug_sample_rows(driver, csv_path, database=DATABASE, sample_size=MAX_DEBUG_SAMPLES):
    print("\n=== DEBUG PRIME RIGHE CSV ===")

    with open(csv_path, mode="r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        with driver.session(database=database) as session:
            for i, row in enumerate(reader):
                if i >= sample_size:
                    break

                print("\n" + "=" * 80)
                print(f"RIGA CSV {i + 1}")

                record = csv_row_to_recommendation_record(row)
                print(f"  RECORD: {record}")

                if record is None:
                    print("  -> Riga non valida")
                    continue

                customer_match = session.run(
                    "MATCH (c:Customer {ids_customer: $id}) RETURN c.ids_customer AS id LIMIT 1",
                    id=record["customerId"],
                ).data()

                level5_match = session.run(
                    "MATCH (l5:Level5 {level5id: $id}) RETURN l5.level5id AS id LIMIT 1",
                    id=record["level5Id"],
                ).data()

                print(
                    f"  Customer match: {'SI' if customer_match else 'NO'} (ids_customer={record['customerId']!r})"
                )
                print(
                    f"  Level5 match:   {'SI' if level5_match else 'NO'} (level5id={record['level5Id']!r})"
                )


# =========================
# Import
# =========================
def load_recommendations_from_csv(driver, csv_path, batch_size=BATCH_SIZE, database=DATABASE):
    total_rows = 0
    valid_rows = 0
    customer_found_rows = 0
    level5_found_rows = 0
    matched_rows = 0

    with open(csv_path, mode="r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        def records():
            nonlocal total_rows, valid_rows
            for row in reader:
                total_rows += 1
                record = csv_row_to_recommendation_record(row)
                if record is not None:
                    valid_rows += 1
                    yield record

        with driver.session(database=database) as session:
            for batch in tqdm(chunked(records(), batch_size), desc="Caricamento RECOMMENDED_TO"):
                batch_stats = session.execute_write(write_recommendation_batch, batch)
                customer_found_rows += batch_stats["customer_found_rows"]
                level5_found_rows += batch_stats["level5_found_rows"]
                matched_rows += batch_stats["matched_rows"]

    print("\n=== RISULTATI IMPORT ===")
    print(f"Righe CSV lette:              {total_rows}")
    print(f"Righe valide processate:      {valid_rows}")
    print(f"Righe con Customer trovato:   {customer_found_rows}")
    print(f"Righe con Level5 trovato:     {level5_found_rows}")
    print(f"Righe con match completo:     {matched_rows}")
    print(f"Righe senza match:            {valid_rows - matched_rows}")


# =========================
# Main
# =========================
def main():
    driver = config.neo4j_driver
    try:
        # Check graph state
        with driver.session(database=DATABASE) as session:
            customer_count = session.run(
                "MATCH (c:Customer) RETURN count(c) AS n"
            ).single()["n"]
            level5_count = session.run(
                "MATCH (l5:Level5) RETURN count(l5) AS n"
            ).single()["n"]

            print(f"Customer presenti: {customer_count}")
            print(f"Level5 presenti:   {level5_count}")

        # Debug
        debug_sample_rows(
            driver,
            CSV_PATH,
            database=DATABASE,
            sample_size=MAX_DEBUG_SAMPLES
        )

        # Import
        load_recommendations_from_csv(
            driver,
            CSV_PATH,
            batch_size=BATCH_SIZE,
            database=DATABASE
        )

        # Final verification
        with driver.session(database=DATABASE) as session:
            total_rec = session.run(
                "MATCH ()-[r:RECOMMENDED_TO]->() RETURN count(r) AS n"
            ).single()["n"]

            print("\n=== VERIFICA FINALE ===")
            print(f"Totale relazioni RECOMMENDED_TO: {total_rec}")

            samples = session.run("""
                MATCH (l5:Level5)-[r:RECOMMENDED_TO]->(c:Customer)
                RETURN c.ids_customer AS customer,
                       l5.level5id AS product,
                       r.product_name AS name
                LIMIT 5
            """).data()

            if samples:
                print("Esempi:")
                for s in samples:
                    print(
                        f"  Level5 {s['product']} -> Customer {s['customer']} ({s['name']})"
                    )
            else:
                print("Nessuna relazione RECOMMENDED_TO trovata.")
    finally:
        config._shutdown()

if __name__ == "__main__":
    main()
