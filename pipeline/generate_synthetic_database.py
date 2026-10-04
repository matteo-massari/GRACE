"""Generate a deterministic, internally consistent synthetic dataset.

The files are deliberately shaped like the records consumed by ``ETL_fact.py``
and ``ETL_recc.py``.  They are suitable for local Neo4j development and do not
contain real customer data.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

SEED = 20260814

FIRST_NAMES = [
    "Alessandro",
    "Andrea",
    "Beatrice",
    "Chiara",
    "Davide",
    "Elena",
    "Federico",
    "Francesca",
    "Giulia",
    "Lorenzo",
    "Marco",
    "Martina",
    "Matteo",
    "Paola",
    "Riccardo",
    "Sara",
    "Simone",
    "Valentina",
]
LAST_NAMES = [
    "Bianchi",
    "Conti",
    "Esposito",
    "Ferrari",
    "Fontana",
    "Gallo",
    "Greco",
    "Leone",
    "Lombardi",
    "Marino",
    "Moretti",
    "Rinaldi",
    "Romano",
    "Russo",
    "Santoro",
    "Serra",
]
LOCATIONS = [
    ("Milano", "MI", "Lombardia", "45.4642", "9.1900"),
    ("Roma", "RM", "Lazio", "41.9028", "12.4964"),
    ("Torino", "TO", "Piemonte", "45.0703", "7.6869"),
    ("Bologna", "BO", "Emilia-Romagna", "44.4949", "11.3426"),
    ("Firenze", "FI", "Toscana", "43.7696", "11.2558"),
    ("Napoli", "NA", "Campania", "40.8518", "14.2681"),
    ("Bari", "BA", "Puglia", "41.1171", "16.8719"),
    ("Padova", "PD", "Veneto", "45.4064", "11.8768"),
]
SECTORS = [("SEC-01", "Hospitality"), ("SEC-02", "Retail"), ("SEC-03", "Industry")]
CHANNELS = [("CH-01", "Direct sales"), ("CH-02", "Distributors"), ("CH-03", "E-commerce")]
SEGMENTS = [("MS-01", "Small businesses"), ("MS-02", "Medium businesses"), ("MS-03", "Large businesses")]
PAYMENT_METHODS = [("PM-01", "Bank transfer"), ("PM-02", "Credit card"), ("PM-03", "Direct debit")]
PAYMENT_TERMS = [("PT-30", "30 days"), ("PT-60", "60 days"), ("PT-90", "90 days")]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot create an empty table: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def product_rows() -> list[dict[str, str]]:
    families = [
        ("Food", ["Canned goods", "Pasta", "Sauces"]),
        ("Beverages", ["Water", "Juices", "Wines"]),
        ("Cleaning", ["Household", "Laundry", "Hygiene"]),
        ("Office", ["Paper", "Stationery", "Filing"]),
        ("Packaging", ["Boxes", "Bags", "Labels"]),
    ]
    rows: list[dict[str, str]] = []
    product_number = 1
    for family_index, (family, categories) in enumerate(families, start=1):
        for category_index, category in enumerate(categories, start=1):
            for variant in range(1, 5):
                product_id = f"PRD-{product_number:03d}"
                rows.append(
                    {
                        "level1id": f"L1-{family_index:02d}",
                        "level1": family,
                        "level2id": f"L2-{family_index:02d}",
                        "level2": f"{family} professional",
                        "level3id": f"L3-{family_index:02d}-{category_index:02d}",
                        "level3": category,
                        "level4id": f"L4-{family_index:02d}-{category_index:02d}",
                        "level4": f"Line {category}",
                        "level5id": product_id,
                        "level5": f"{category} selection {variant}",
                    }
                )
                product_number += 1
    return rows


def salesrep_rows(rng: random.Random) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for index in range(1, 31):
        first_name, last_name = rng.choice(FIRST_NAMES), rng.choice(LAST_NAMES)
        city, _, region, _, _ = LOCATIONS[(index - 1) % len(LOCATIONS)]
        district = f"DST-{((index - 1) % 10) + 1:02d}"
        territory = f"TER-{((index - 1) % 5) + 1:02d}"
        macro_region = (
            "MR-NORTH" if region in {"Lombardia", "Piemonte", "Veneto", "Emilia-Romagna"} else "MR-CENTRAL-SOUTH"
        )
        manager_id = f"SR-{((index - 1) // 10) * 10 + 1:03d}"
        channel_id, channel_description = CHANNELS[(index - 1) % len(CHANNELS)]
        sector_id, sector_description = SECTORS[(index - 1) % len(SECTORS)]
        rows.append(
            {
                "ids_salesrep": f"SR-{index:03d}",
                "dsc_salesrepfirstname": first_name,
                "dsc_salesrepsurname": last_name,
                "att_mobilephonenumber": f"+39 3{20 + index:02d} 555 {index:04d}",
                "att_email": f"{first_name.lower()}.{last_name.lower()}.{index}@example.test",
                "att_street": f"Commercial Street {index}",
                "att_zipcode": f"{20100 + index:05d}",
                "att_cityname": city,
                "att_saleschannel": channel_id,
                "dsc_channel": channel_description,
                "att_sector": sector_id,
                "dsc_sector": sector_description,
                "att_marketsegment": SEGMENTS[(index - 1) % len(SEGMENTS)][0],
                "att_businessarea": "North" if macro_region == "MR-NORTH" else "Central-South",
                "dsc_job": "Account Manager" if index % 3 else "Sales Specialist",
                "ids_district": district,
                "dsc_district": f"District {district[-2:]}",
                "ids_districtsalesrep": manager_id,
                "dsc_districtsalesrepfirstname": "Manager",
                "dsc_districtsalesrepsurname": district[-2:],
                "ids_territory": territory,
                "dsc_territory": f"Territory {territory[-2:]}",
                "ids_territorysalesrep": manager_id,
                "dsc_territorysalesrepfirstname": "Manager",
                "dsc_territorysalesrepsurname": territory[-2:],
                "ids_mregion": macro_region,
                "dsc_mregion": (
                    "Northern macroregion" if macro_region == "MR-NORTH" else "Central-southern macroregion"
                ),
                "ids_mregionsalesrep": "SR-001" if macro_region == "MR-NORTH" else "SR-011",
                "dsc_mregionsalesrepfirstname": "Manager",
                "dsc_mregionsalesrepsurname": "Regional",
            }
        )
    return rows


def customer_rows(rng: random.Random) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for index in range(1, 101):
        city, county, _, latitude, longitude = rng.choice(LOCATIONS)
        sector_id, sector_description = rng.choice(SECTORS)
        channel_id, channel_description = rng.choice(CHANNELS)
        segment_id, segment_description = rng.choice(SEGMENTS)
        payment_method, payment_method_description = rng.choice(PAYMENT_METHODS)
        payment_terms, payment_terms_description = rng.choice(PAYMENT_TERMS)
        company = f"{rng.choice(LAST_NAMES)} {rng.choice(['S.r.l.', 'S.p.A.', 'S.n.c.'])}"
        rows.append(
            {
                "ids_customer": f"C-{index:03d}",
                "dsc_customer": company,
                "partita_iva": f"IT{10000000000 + index}",
                "att_numberemployees": str(rng.randint(2, 500)),
                "att_potential": rng.choice(["Low", "Medium", "High"]),
                "att_creditlimit": str(rng.choice([5000, 10000, 25000, 50000, 100000])),
                "att_payterms": payment_terms,
                "att_paytermsdescription": payment_terms_description,
                "att_paymentmethod": payment_method,
                "att_paymentmethoddescription": payment_method_description,
                "att_email": f"purchasing.{index:03d}@example.test",
                "att_street": f"Business Street {index}",
                "att_zipcode": f"{10000 + index:05d}",
                "att_longitude_x": longitude,
                "att_latitude_y": latitude,
                "att_sector": sector_id,
                "att_sectordescription": sector_description,
                "att_branch": f"BR-{((index - 1) % 8) + 1:02d}",
                "att_branchdescription": f"Branch {((index - 1) % 8) + 1:02d}",
                "att_saleschannel": channel_id,
                "dsc_channel": channel_description,
                "att_marketsegment": segment_id,
                "att_marketsegmentdescription": segment_description,
                "att_city": city,
                "att_county": county,
                "att_country": "IT",
            }
        )
    return rows


def purchase_rows(
    rng: random.Random, customers: list[dict[str, str]], products: list[dict[str, str]]
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    start = date(2025, 8, 15)
    for customer_index, customer in enumerate(customers):
        # Deliberately event-level purchases: each consumer has 10--100 rows.
        number_of_purchases = rng.randint(10, 100)
        favourite_family = customer_index % 5
        preferred_products = products[favourite_family * 12 : favourite_family * 12 + 12]
        other_products = [product for product in products if product not in preferred_products]
        # Keep at least ten products unseen per customer for the placeholder top-10.
        purchasable_products = preferred_products + rng.sample(other_products, k=38)
        for _ in range(number_of_purchases):
            product = rng.choice(preferred_products if rng.random() < 0.72 else purchasable_products)
            rows.append(
                {
                    "ids_customer": customer["ids_customer"],
                    "level5id": product["level5id"],
                    "dat": (start + timedelta(days=rng.randint(0, 364))).isoformat(),
                    "quantity": str(rng.randint(1, 12)),
                    "unit_price": f"{rng.uniform(4.5, 250.0):.2f}",
                }
            )
    return sorted(rows, key=lambda row: (row["dat"], row["ids_customer"], row["level5id"]))


def recommendation_rows(
    customers: list[dict[str, str]], products: list[dict[str, str]], purchases: list[dict[str, str]]
) -> list[dict[str, str]]:
    purchased: dict[str, set[str]] = defaultdict(set)
    for purchase in purchases:
        purchased[purchase["ids_customer"]].add(purchase["level5id"])

    rows: list[dict[str, str]] = []
    for customer in customers:
        candidates = [p for p in products if p["level5id"] not in purchased[customer["ids_customer"]]]
        # There are always at least ten candidates because customers buy from a
        # preferred subset; nevertheless retain the guard if input parameters change.
        for rank, product in enumerate(candidates[:10], start=1):
            rows.append(
                {
                    "ids_customer": customer["ids_customer"],
                    "ids_product": product["level5id"],
                    "product_name": product["level5"],
                    "rank": str(rank),
                    "score": f"{1 - (rank - 1) * 0.06:.2f}",
                    "source": "placeholder_not_gnn",
                }
            )
    return rows


def create_gnn_tables(
    output_dir: Path,
    customers: list[dict[str, str]],
    products: list[dict[str, str]],
    purchases: list[dict[str, str]],
    recommendations: list[dict[str, str]],
) -> None:
    dataset_dir = output_dir / "gnn" / "training_tables" / "dataset_None_4_12"
    write_csv(
        dataset_dir / "basket_data.csv",
        [
            {"ids_customer": p["ids_customer"], "ids_product": p["level5id"], "dat": p["dat"]}
            for p in purchases
        ],
    )
    write_csv(dataset_dir / "product.csv", [{"ids_product": p["level5id"]} for p in products])
    write_csv(
        dataset_dir / "product_names.csv",
        [{"ids_product": p["level5id"], "product_name": p["level5"]} for p in products],
    )
    write_csv(
        dataset_dir / "customer.csv",
        [
            {
                "ids_customer": c["ids_customer"],
                "sector": c["att_sector"],
                "potential": c["att_potential"],
                "county": c["att_county"],
                "creditlimit": c["att_creditlimit"],
            }
            for c in customers
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the synthetic Neo4j/GNN dataset.")
    parser.add_argument("--output-dir", type=Path, default=Path("database"))
    args = parser.parse_args()

    rng = random.Random(SEED)
    products = product_rows()
    customers = customer_rows(rng)
    salesreps = salesrep_rows(rng)
    customer_salesreps = [
        {"ids_customer": c["ids_customer"], "ids_salesrep": f"SR-{((index * 7) % 30) + 1:03d}"}
        for index, c in enumerate(customers)
    ]
    purchases = purchase_rows(rng, customers, products)
    recommendations = recommendation_rows(customers, products, purchases)

    write_csv(args.output_dir / "customers.csv", customers)
    write_csv(args.output_dir / "salesreps.csv", salesreps)
    write_csv(args.output_dir / "products.csv", products)
    write_csv(args.output_dir / "customer_salesreps.csv", customer_salesreps)
    write_csv(args.output_dir / "purchases.csv", purchases)
    write_csv(args.output_dir / "recommendations_placeholder.csv", recommendations)
    create_gnn_tables(args.output_dir, customers, products, purchases, recommendations)

    purchase_counts = defaultdict(int)
    for purchase in purchases:
        purchase_counts[purchase["ids_customer"]] += 1
    print(f"Dataset created in: {args.output_dir.resolve()}")
    print(f"Customers: {len(customers)} | Products: {len(products)} | Sales representatives: {len(salesreps)}")
    print(
        "Purchase events: "
        f"{len(purchases)} (min/max per customer: "
        f"{min(purchase_counts.values())}/{max(purchase_counts.values())})"
    )
    print(f"Raccomandazioni placeholder: {len(recommendations)}")


if __name__ == "__main__":
    main()
