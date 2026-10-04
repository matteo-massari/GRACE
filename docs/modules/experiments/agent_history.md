# Performance Model

> Historical notes: examples, run and parameters refer to experiments
> described, not to current startup instructions.

> Anonymized version: names, identifiers, geographic area, hierarchical codes and product codes have been replaced with consistent fictitious labels. Sectors, categories and quantitative results have been maintained to preserve the value of the analysis.


1. Evaluation of requests based on counts and first movements on the graph
To test model behavior on count-based requests, we use the following question:

**Question**: What is the best seller assigned to the A geographical area in the Metal sector?
**Answer**:
```md
**Result:** the best sales representative in Geographic Area A in the Metal sector is **SELLER A**, ID **VEND-001**.
**Evidence:** manages **137 distinct customers**, the highest count among representatives linked to Geographic Area A and the Metal sector.
**Next step:** use this representative as a benchmark for comparing portfolios and opportunities in the territory.
```

Note
The question is not a real case of use, but it serves as an internal test to verify whether the model is able to correctly identify a reference subject for interaction.
In the absence of direct metrics on performance, we consider that with the greatest number of customers in the Metal sector in the A geographic area.


2. We impersonate the seller and ask for an information overview
**Question**: Hi I am Seller A Let me have an overview on the customer base, on my operating sector and on my hierarchical scale

**Answer**:
```md
**Result** → Hello Seller A: your ID is **VEND-001**. You work in **Geographic Area A** in the **Metal** sector, with **137 customers** and opportunities in fasteners, tools, and metalworking.

**Evidence**
- **Customer base:** 137 assigned customers.
- **Sector mix:**
  - Metal: **88 customers**
  - Installations: 16
  - Wood: 14
  - Construction: 13
  - Cargo: 6
- **Main segments:**
  - Metal sector – construction: **46 customers**
  - Maintenance: 26
  - Metal sector – manufacturing: 16
  - Construction companies / large building sites: 10
- **Channels:** 59 HW customers on channel 3, 52 HW on channel 2, 23 HW on channel 4, and 3 PO.
- **Most represented branches:**
  - Metal window manufacturing, code **CAT-001**: 22 customers
  - Metal/steel construction, code **CAT-002**: 17
  - Equipment manufacturing, code **CAT-003**: 10
  - Food and beverage production, code **CAT-004**: 8
- **Most common Level5 categories:**
  - Neutral window sealant, code **PROD-001**: 45 customers
  - DIN 934 galvanized steel nut, code **PROD-002**: 36
  - Plus Quadro-L percussion drill bit, code **PROD-003**: 32
  - ISO 4017 galvanized 8.8 steel, code **PROD-004**: 31
  - DIN 522 galvanized steel, code **PROD-005**: 31
  - PH C 6.3 bit, code **PROD-006**: 29
  - Rubber-coated tape measure, code **PROD-007**: 29
- **Customers with the broadest purchase range:**
  - **Customer 001**, ID **CLI-001**: 262 categories
  - **Customer 002**, ID **CLI-002**: 241
  - **Customer 003**, ID **CLI-003**: 198
  - **Customer 004**, ID **CLI-004**: 170
  - **Customer 005**, ID **CLI-005**: 153
- **Available hierarchy:**
  - City/area: **Geographic Area A**
  - Sector: **Metal**
  - District: **District D-01**
  - Job: **JOB-001**
  - Territory and macroregion: not populated for this profile.

**Next step** → Start with metal-sector customers who buy fasteners or tools, but not both. Combine screws, bits, drills, and cutting discs. Prioritize **Customer 001**, **Customer 002**, and **Customer 003**, who have the broadest purchase histories.
```

3) We introduce Recommendations

**Question** Today I have to work in the Metal industry, tell me what are the 5 most recommended products that allow me to cover more customers than that industry in just one lap

**Answer**:
```md
**Result** → To reach more **Metal** sector customers in one sales round, start with these five recommended products:

1. **DIN 522 galvanized steel** — code **PROD-005**: **64 customers**
2. **Utility knife with plastic handle and slider** — code **PROD-008**: **60 customers**
3. **DIN 934 galvanized I6I / I8I steel** — code **PROD-002**: **58 customers**
4. **ISO 4017 galvanized 8.8 steel** — code **PROD-004**: **50 customers**
5. **HSS DIN338 RN 130° twist drill** — code **PROD-009**: **50 customers**

**Evidence**
- Counts refer to distinct **Metal** sector customers assigned to Seller A.
- These products have recommendation relationships in the database.
- These counts measure potential reach per product. Customers may appear in several categories, so the counts cannot be added.

**Next step** → Start the sales round with **DIN 522**, **utility knives**, and **DIN 934**, the three categories with the widest reach. Add **ISO 4017** and **HSS drills** as complementary products.
```

## 4

**Question** Which customers can I recommend all 5 of these products?

**Answer**:
```md
**Result** → There are **46 Metal-sector customers** linked to all five products through purchases and/or recommendations. Prioritize customers who have purchased one to four of them for cross-selling.

**Evidence**

**High priority – purchased 1 product**
- **Customer 006** — ID **CLI-006**
- **Customer 007** — ID **CLI-007**
- **Customer 008** — ID **CLI-008**
- **Customer 009** — ID **CLI-009**
- **Customer 010** — ID **CLI-010**
- **Customer 011** — ID **CLI-011**
- **Customer 012** — ID **CLI-012**
- **Customer 013** — ID **CLI-013**

**High priority – purchased 2 products**
- **Customer 014** — ID **CLI-014**
- **Customer 015** — ID **CLI-015**
- **Customer 010** — ID **CLI-016**
- **Customer 017** — ID **CLI-017**
- **Customer 018** — ID **CLI-018**

**Medium priority – purchased 3 products**
- **Customer 005** — ID **CLI-005**
- **Customer 019** — ID **CLI-019**
- **Customer 020** — ID **CLI-020**
- **Customer 021** — ID **CLI-021**
- **Customer 022** — ID **CLI-022**
- **Customer 023** — ID **CLI-023**

**Medium priority – purchased 4 products**
- **Customer 024** — ID **CLI-024**
- **Customer 025** — ID **CLI-025**
- **Customer 026** — ID **CLI-026**
- **Customer 026** — ID **CLI-026**
- **Customer 021** — ID **CLI-021**

**Customers who already purchased all five**
- **Customer 003** — ID **CLI-003**
- **Customer 004** — ID **CLI-004**
- **Customer 027** — ID **CLI-027**
- **Customer 028** — ID **CLI-028**
- **Customer 029** — ID **CLI-029**

**Customers linked to all five products with no recorded purchases of them**
- **Customer 030** — ID **CLI-030**
- **Customer 031** — ID **CLI-031**
- **Customer 032** — ID **CLI-032**
- **Customer 033** — ID **CLI-033**
- **Customer 034** — ID **CLI-034**
- **Customer 035** — ID **CLI-035**
- **Customer 036** — ID **CLI-036**
- **Customer 037** — ID **CLI-037**
- **Customer 038** — ID **CLI-038**
- **Customer 039** — ID **CLI-039**
- **Customer 040** — ID **CLI-040**
- **Customer 041** — ID **CLI-041**
- **Customer 042** — ID **CLI-042**
- **Customer 043** — ID **CLI-043**
- **Customer 044** — ID **CLI-044**
- **Customer 045** — ID **CLI-045**
- **Customer 046** — ID **CLI-046**
- **Customer 047** — ID **CLI-047**
- **Customer 048** — ID **CLI-048**

Note: **Customer 026** and **Customer 021** are duplicated in the original list; count each only once.

**Next step** → Build the round around **Customer 006, Customer 007, Customer 008, Customer 011, Customer 012, Customer 013, Customer 005, and Customer 021**. Their existing purchases leave room to complete the range. Bring the five products and propose the full package.
```

5)

**Question** We analyze the Customer 031: what are your most purchased categories?

**Answer**:
```md
**Result** → For **Customer 031**, ID **CLI-031**, no category leads by purchase count: all recorded categories have **1 purchase/connection** in the database.

**Evidence**
Purchased categories include:
- **X-300 banded earplugs** — code **PROD-010**
- **Basic workshop cabinet** — code **PROD-011**
- **1/4-inch socket set, 33 pieces** — code **PROD-012**
- **Round-shaft slotted screwdriver** — code **PROD-013**
- **Short combination wrench** — code **PROD-014**
- **KBL1 PA cable tie with plastic tongue** — code **PROD-015**
- **PTFE thread-sealing tape** — code **PROD-016**
- **ZEBRA DIN ISO 5746 combination pliers** — code **PROD-017**
- **Short combination wrench set, 11 pieces** — code **PROD-018**
- **Universal steel hand brush** — code **PROD-019**

Recorded purchase dates extend through **May 2026**, for example for **X-300 banded earplugs** and **Replacement pads for X-300 banded earplugs**.

**Next step** → Do not base the visit on a “most purchased” category. Consolidate supply across hand tools, PPE, cable ties, and cleaning/maintenance products. Purchases span many families, creating room for a broader proposal.
```

6) Cross sell and reasoning analysis on products

**Question**: excellent now you do a cross-sell analysis on the 5 products recommended for Customer 031: identify the related products, for example by category, and return the best sales strategies to offer to the customer.

**Answer**:
```md
**Result** → Customer 031 has purchased products related to **3 of the 5 targets**. The strongest cross-selling opportunities are **cutting tools**, **material processing**, and **fasteners**. All five targets are recommended, but **none has been purchased**.

**Evidence**

| Recommended product | Status | Category | Supporting evidence |
|---|---|---|---|
| **DIN 522 galvanized steel** — **PROD-005** | Recommended, not purchased | Level2 **Washers** — Level1 **Fasteners** | No other purchases in the same Level2 category |
| **Utility knife with plastic handle and slider** — **PROD-008** | Recommended, not purchased | Level2 **Cutting tools** — Level1 **Hand tools** | Already purchased two products in this category: **Combination folding pocket knife** — **PROD-020**, last purchased October 2025; **Martor Secunorm Smartcut safety knife** — **PROD-021**, last purchased November 2025 |
| **DIN 934 galvanized I6I/I8I steel** — **PROD-002** | Recommended, not purchased | Level2 **Nuts** — Level1 **Fasteners** | No purchases in the same Level2 category |
| **ISO 4017 galvanized 8.8 steel** — **PROD-004** | Recommended, not purchased | Level2 **Metric/imperial screws** — Level1 **Fasteners** | Already purchased **DIN 965 galvanized 4.8 steel H** — **PROD-022**, last purchased March 2026 |
| **HSS DIN338 RN 130° twist drill** — **PROD-009** | Recommended, not purchased | Level2 **Drilling and chiseling bits** — Level1 **Material processing** | Already purchased **DUO-S cylindrical percussion drill bit set, 10 pieces, Syska** — **PROD-023**, last purchased November 2025 |

**Sales strategies**

1. **Start with the utility knife: the strongest cross-selling opportunity.**
   The customer has purchased two cutting products. Offer the **Utility knife with plastic handle and slider** as a more structured alternative or a standard product for recurring tasks.
   **Sales hypothesis:** focus on replacement, immediate availability, and standardized equipment.

2. **Introduce professional drilling products.**
   Already purchased a percussion drill set; offer the **HSS DIN338 drill** for a different drilling requirement.
   **Sales hypothesis:** position it as a complement for metalworking, without claiming it replaces the existing drills.

3. **Build a complete fastening kit.**
   The customer has purchased a **DIN 965**, so you can offer **ISO 4017**, **DIN 934** and **DIN 522** to expand the kit: screw, nut, and washer.
   **Sales hypothesis:** a kit may be more effective than three separate items, although the database does not demonstrate specific nut or washer consumption.

4. **Use fasteners to complete the proposal rather than as the initial priority.**
   **ISO 4017** has the closest link to the recorded DIN 965 purchase. Then introduce **DIN 934** and **DIN 522** to complete the fastening system.

**Next step** → During the visit, first propose **utility knife + HSS drill**, supported by recorded purchases. Then offer an **ISO 4017 + DIN 934 + DIN 522 kit**. Start from existing purchases to ground the proposal in evidence.
```
