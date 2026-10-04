# Theory

In this section we address the theory behind what made this project possible. Starting from the information on a graph structure we reach the generation of a new information.

## Introduction to Knowledge Graphs

**Knowledge Graphs (KG)** are semantic networks that represent relationships between entities.
From the structural point of view, they are based on **graphs**, a discrete mathematical structure that, through nodes and edges, allows to model objects and relationships of the real world.

These structures allow to represent information that may initially appear disconnected, forming articulated networks of interactions and semantic dependencies.

Knowledge Graph can be stored in **Graph Database** as **Neo4j** or questioned by standard as **RDF** and **SPARQL**, which allow to represent and interrogate relational data in semantic form.

A definition of Knowledge Graph is as follows:

> *"A graph of data intended to accumulate and convey knowledge of the real world, whose nodes represent entities of interest and whose edges represent relations between these entities."*

## Types of Knowledge Graphs

Knowledge Graphs can be shaped according to different paradigms, depending on the type of data and formalism chosen. Among the main models are:

- Directed Edge-Labelled Graph
- Heterogeneous Graph
- Property Graph

### Directed Edge-Labelled Graph

**Directed Edge-Labelled Graphs** are graphs in which:

- **Nodes** represent objects, entities or concepts;
- **Edges** connect the nodes and are characterized by:
  - a **direction**, indicating the orientation of the relationship;
  - a **label**, describing the type of relationship between the two nodes.

From the mathematical point of view, a directed edge-labelled graph can be seen as a traditional direct graph:

```{math}
G = (U, L)
```

where:

- {math}`U` is the set of nodes;
- {math}`L` is the set of direct edges.

This structure adds a label for each edge {math}`\left(u_i, u_j\right) \in L`, obtaining an extended set of edges:

```{math}
L^* = \left\{(u_i, u_j, a_{ij}) \mid (u_i, u_j) \in L \right\}
```

As part of the representation of knowledge, this type of graph is fundamental because it is the basis of the **Resource Description Framework (RDF)**.

The **Resource Description Framework** is a standard model for data coding in the semantic Web.
Its structure facilitates the integration of data from different patterns and supports the evolution of the scheme over time without requiring invasive structural changes.

In RDF, the information is represented through **triples** in the form:

```{math}
(s, p, o)
```

where:

- $s$ = subject, i.e. the starting entity;
- $p$ = predicate, that is, the relationship;
- $o$ = object, i.e. the target entity or the associated value.

```{figure} ../images/Directed-edge-labelled-graph.png
:alt: Directed edge-labelled graph
:width: 70%
:align: center

Example of a directed edge-labelled graph.
```

In other words, a RDF triplet can be interpreted as a directed, labeled edge:

```{math}
\text{subject} \xrightarrow{\text{predicate}} \text{object}
```

For this reason, a **RDF store** can be seen as a collection of reports expressed in the form of direct labeled graph.

The standard language used to question a RDF triple store is **SPARQL**, which allows you to define search patterns on RDF triplets and retrieve information on the basis of the reports present in the graph.

### Property Graph

**Property Graphs** are also composed of nodes and edges, but with a fundamental difference: both nodes and edges can possess **property**, expressed as key-value pairs.

Each node has:

- a unique identifier;
- a set of outgoing edges;
- a set of incoming edges;
- a collection of property.

Each edge has:

- a unique identifier;
- a node source;
- a node destination;
- a text label describing the relationship;
- a collection of property.

Example:

```text
(1: {name: "Alice", age: 32}) -[:knows {type: "friend"}]-> (2: {name: "Bob", age: 35})
```

```{figure} ../images/property_graph.png
:alt: Property graph
:width: 70%
:align: center

Example of a property graph.
```

One of the best known databases that implement this paradigm is **Neo4j**, developed in Java.
Neo4j allows you to store nodes and labeled relationships, enriched with properties, and is designed to make the graph crossing natural.

The query language typically used in this context is **Cypher**.

## Knowledge Representation and Reasoning

The **Knowledge Representation and Reasoning (KRR)** is a branch of artificial intelligence that is responsible for representing knowledge in a structured way, so as to allow computer systems to make decisions and make inferences.

Knowledge Graphs are particularly suitable for this purpose because they represent concepts, entities and real world relations in a clear, structured and interpretable manner.

### Knowledge Representation

In the construction of a representation of knowledge it starts from some fundamental **assumptions**:

- **Closed World Assumption (CWA)**: the graph is considered complete; what is not present is considered false.
- **Open World Assumption (OWA)**: the graph may be incomplete; what is not present is not necessarily false.

A central concept is that of **ontology**, i.e. a formal and shared representation of knowledge of a domain.

Its main components are:

- individuals;
- property;
- classes;
- axioms and restrictions.

### Knowledge Reasoning

The reasoning on knowledge can take place mainly in two forms:

- **deduction**: start from general rules to derive particular cases;
- **induction**: it starts from special observations to formulate generalizations.

Example of deduction:

> All men are mortal.
> Socrates is a man.
> So Socrates is mortal.

Example of induction:

> I have observed many white swans.
> I can assume the swans are white.

In the context of Knowledge Graph:

- **Deductive Knowledge**: is to derive new truths from what is already known, through rules of inference.
- **Inductive Knowledge**: it is to derive new truths from patterns observed in data, typically through machine learning approaches.

These two approaches are not excluded: the graph remains the same, but the techniques used to exploit it change.

```{figure} ../images/ded_ind.jpg
:alt: Deductive and inductive knowledge
:width: 70%
:align: center


Difference between deductive and inductive knowledge.
```

#### Deductive Knowledge

Deductive knowledge concerns facts not explicitly stored in the graph but derived logically from ontology or formal rules.

In the reasoning on complete ontologies often emerges a problem of **undecidability**: there is no algorithm that, for any ontology in input, always ensures a correct answer in finite time.
For this reason, formalities are adopted in practice that balance **expressive** and **computational complexity**.

The main strategies for deductive reasoning are:

- **Rules**
- **Description Logics (DL)**

##### Rules

The rules typically have the form:

> [condition] [conclusion]

They allow to deduce new knowledge by applying predefined logic rules, such as:

- symmetry: if A knows B, then B knows A;
- If A is part of B and B is part of C, then A is part of C.

##### Description Logics

The **Description Logics** are formalisms to represent knowledge through:

- **individuals**: objects of the world;
- **classes** or **concepts**: sets of individuals;
- **properties** or **roles**: relationships between individuals.

A knowledge base in Description Logic is generally composed of:

- **TBox (Terminological Box)**: contains general knowledge, i.e. axioms, hierarchy and relationships between concepts;
- **ABox (Assertional Box)**: contains specific facts about individuals;
- **RBox (Role Box)**: contains axioms on roles or properties.

This separation is useful both from the computational point of view and for the reuse of knowledge.

The main compromise in Description Logics is to limit some constructs to keep reasoning decidable.

In general:

- in contexts **CWA** you tend to prefer approaches based on **rules**;
- in contexts **OWA** you tend to prefer approaches based on **Description Logics**.

Basically:

- **Rules** transform existing facts into new facts;
- **Description Logics** allow a more structured reasoning on the whole knowledge base.

#### Inductive Knowledge

Unlike deductive knowledge, inductive knowledge starts from concrete observations to generalize.
This makes it particularly suitable for approaches **machine learning**, supervised or self-supervised.

The main approaches to inductive knowledge are:

- **Numerical Learning**
- **Symbolic Learning**

##### Numerical Learning

Produces a numerical representation of knowledge, transforming entities and relationships into vectors or embedding.

Among the main approaches we find:

- **Graph Analytics**
  Analyze the topology of the graph without necessarily using semantic labels. Measures such as centrality, clustering coefficient and PageRank describe the role of nodes in the network.

- **Embedding Extraction**
  Excerpt dense numerical representations of the graph.

In this area they fall:

- **Knowledge Graph Embeddings (KGE)**, typically self-supervised;
- **Graph Neural Networks (GNN)**, typically supervised or task-driven.

The main difference is that:

- **KGE** explicitly model triplets and relationships;
- **GNNs** model the structure of the graph and the propagation of information between nodes.

Both produce **embedding** can then be used in downstream task as:

- node classification;
- regression;
- link prediction;
- graph classification;
- recommendation.

##### Symbolic Learning

It produces symbolic and interpretable representations of knowledge, remaining in the logical-formal domain.

Among the main approaches:

- **Rule Mining**
  It extracts logical rules from statistical patterns observed in the graph.

- **Axiom Mining**
  Ontological axioms, such as assumptions between classes or domain and range constraints, based on observed relationships.

## Knowledge Graph Embedding

**Knowledge Graph Embeddings** learn vector representations of entities, relationships and facts from existing triples, with the aim of capturing the plausibility of a triple {math}`(s, p, o)`.
The main components of these models are:

1. **Representation Space**: the vector space in which entities and relationships are represented;
2. **Scoring Function**: the function that assigns a score to a triple;
3. **Encoding Model**: how entities and relationships interact in embedding space;
4. **Additional information**: constraints, attributes, or external signals.

The training process typically follows these steps:

1. random initialization of vectors of entities and relationships;
2. selection of batches of real triples;
3. generation of negative triplets altering entities or relationships;
4. Scoring function optimization up to convergence.

Models like **TransE**, **RotatE** and **ComplEx** optimize a function that assigns high scores to true triplets and low scores to false ones.

## Graph Neural Networks

Unlike KG Embeddings, which produce static representations, the **Graph Neural Networks (GNN)** produce **contextual** embeddings and **task-specific**.

The representation of a node is built by aggregating the information of its neighbors, and the whole process is optimized compared to the final task.

A GNN can be seen as a neural transformation that works on:

- attributes of nodes;
- attributes of the edges;
- optional global context;

preserving the structural symmetries of the graph, such as invariance with the permutation of nodes.

### Architecture

```{figure} ../images/gnn.png
:alt: Graph neural network architecture
:width: 75%
:align: center

General structure of a Graph Neural Network.
```

#### Input

A Graph Neural Network takes input:

- the initial features of the nodes;
- the structure of the graph;
- possibly feature on the edges;
- possibly global information on the graph.

To represent the graph, instead of using dense matrices, you often use structures scattered as lists of adjacency.
This prevents unnecessary computations on the disconnected parts of the graph.

Formally, we can consider:

- **nodes** {math}`V`;
- **edges** {math}`E`;
- **global context** {math}`U`.

#### GNN Block

In a simple GNN:

- each node passes through a dedicated neural module;
- each edge passes through a dedicated module;
- the global context can also be updated.

The result is a new graph with the same structure but with updated embedding.

#### Message Passing

The **message passing** is the central mechanism of the GNN: it allows the nodes to communicate with each other through the structure of the graph.

The process is divided into three stages:

1. creating the message;
2. aggregation of received messages;
3. update of the node representation.

```{figure} ../images/graph_message_passing.svg
:alt: Graph message passing
:width: 80%
:align: center

Structure of the message-passing process.
```

The three main functions are:

1. **Message Function**: builds messages to send to the neighbors;
2. **Aggregation Function**: combines messages from neighbors;
3. **Update Function**: updates the representation of the node.

Common aggregations include:

- sum;
- mean;
- maximum;
- attention.

A general formulation of the update is:

```{math}
h_v^{(k+1)} = f\left(
h_v^{(k)},
\mathrm{Agg}\left(\left\{ h_u^{(k)} : u \in \mathcal{N}(v) \right\}\right)
\right)
```

where:

- {math}`h_v^{(k)}` is the representation of node {math}`v` at layer {math}`k`;
- {math}`\mathcal{N}(v)` is the set of neighbors of {math}`v`;
- {math}`\mathrm{Agg}` is the aggregation function;
- {math}`f` is the update function.

In this sense, the new representation of the node is precisely the vector {math}`h_v^{(k+1)}`.

### Prediction Tasks for GNNs

GNN can be used for several tasks:

- **Graph-level task**: prediction of ownership of the entire graph;
- **Node-level task**: prediction of labels or roles of nodes;
- **Edge-level task**: prediction on pairs of nodes or edges.

To switch from local representations to an aggregate representation, operators often use **pooling**.

The pooling typically follows two phases:

1. collection of embeddings in a matrix;
2. aggregation through a reduction to a single embedding.

We can therefore make forecasts on:

- Node: MLP on node embedding;
- arc: combination of embedding of the two nodes accidents;
- graph: global pooling followed by ranking.

### Graph Neural Networks vs Knowledge Graph Embeddings

| Knowledge Graph Embeddings | Graph Neural Networks |
|---|---|
| static embeddings | contextual embeddings |
| global representations | representations built locally via neighborhood |
| less dependent on the final task | directly optimized for the task |
| focus on triples and relationships | focus on local and global graph structure |

## Enrichment of a Graph Database

The **knowledge graph enrichment** is the process of expanding and improving a pre-existing graph through:

- addition of new data;
- refinement of connections;
- improvement of the structure.

This process can start from:

- **text**
- **structured data**

### Enrichment from Text

In the case of texts, a typical pipeline includes:

1. **Pre-processing**
   tokenization, part-of-speech tagging, dependency parsing, word sense disambiguation;
2. **NER (Named Entity Recognition)**
   identification of appointed entities;
3. **Entity Linking**
   connection of existing nodes in the graph;
4. **Relation Extraction**
   extraction of new relationships between entities.

### Enrichment from Structured Data

In the case of table or structured data:

- **Direct Mapping**
  the system automatically generates a graph from the table;
- **Custom Mapping**
  custom rules are defined, for example with languages as **R2RML**, to transform rows and columns into joint nodes and edges with an existing graph.

Alignment can be achieved by query or pipeline **ETL** and mechanisms of **CDC (Change Data Capture)**.

## Enterprise Knowledge Graph

The construction of Knowledge Graph company enables many concrete applications.
Among the typical features are:

- data from mixed, internal and external sources;
- large dimensions;
- continuous refinement;
- constant update;
- integration between ontologies and machine learning;
- frequent use of lightweight ontologies.

Knowledge Graph makes relationships **explicit** and central in the data model.
This allows you to navigate in a natural way complex, heterogeneous and scattered connections.

Among the most common problems faced in the company are:

- node classification;
- graph or subgraph classification;
- clustering;
- link prediction;
- identification of the most influential nodes.

Main business applications include:

- mapping of business knowledge;
- improvement of research skills;
- recommendation system;
- Conversational agents and GraphRAG systems;
- traceability of events and processes.

## Agentic AI and Graph Databases

The main applications of the Graph Database in combination with intelligent systems include:

- **GraphRAG**
- long-term relational memory
- reasoning on complex structures

### Graph Retrieval-Augmented Generation

A frequent limit of LLM in retrieval tasks concerns the difficulty of working with structured data, understanding hierarchical relationships and maintaining consistency in multi-hop scenarios.

> **Multi-hop**: In a traditional RAG the model must connect multiple text fragments. After a few steps, the context can degrade. A graph instead allows to express and preserve such relationships.

The **Graph Retrieval-Augmented Generation (GraphRAG)** incorporates structured graph data, organizing information in terms of:

- entity;
- relations;
- attributes.

The central idea is that the graph outlines part of structural reasoning, reducing cognitive load on the linguistic model.

### GraphRAG Architecture

A GraphRAG pipeline can follow these steps:

1. the user asks a question that implies relational knowledge;
2. the system extracts entities and relationships from the query;
3. the request is translated into a query on the graph, for example in Cypher;
4. a retriever extracts data from the graph database through traversal, semantic signals or embedding models;
5. an orchestrator refines the context;
6. the generator produces the final answer.

## Graph Learning Recommendation Systems

The use of **graph learning** in the recommendations system is natural because the interaction data are intrinsically graphs:

- users who interact with products;
- products shared between users;
- multihop relationships between behaviors and attributes.

This makes graph learning techniques very natural for:

- build ranking based recommendation systems;
- integrate side information;
- model multi-hop dependencies.

| Classical Collaborative Filtering | Graph Learning on Graphs |
|---|---|
| works on scattered user-item matrices | directly represents interactions as a graph |
| suffers from cold start | can mitigate cold start with side information |
| scale with difficulty on complex relationships | naturally models multi-hop dependencies |
| limited integration of extra information | easily integrates additional features |
| often oriented toward explicit scores | naturally suited to top-k recommendations |

### Input

Recommendation systems on graph generally work with three data categories:

- **General interaction data**: non sequential user-item interactions;
- **Sequential interaction data**: temporal interaction sequences;
- **Side information**: user attributes, item attributes, social graph, ontologies, knowledge graph.

### Information Propagation

The key concept that improves the quality of the recommendation is **dissemination of information**.

Example:

- user A saw the movie X;
- the movie X was also seen by user B;
- user B also saw the film Y.

Although user A did not see Y, the graph suggests a possible relationship between A and Y.
This ability to capture connections **multi-hop** is one of the main advantages of graph learning compared to classic models.

### Output

The end nodes embedding incorporates the information propagated through the graph.
These embeddings are then passed to a downstream module that produces the final recommendation.

## Approaches to Graph Learning Recommendation Systems

```{figure} ../images/GLRS_methods.png
:alt: Graph learning recommendation system methods
:width: 80%
:align: center

Overview of the main approaches to graph learning recommendation systems.
```

### Random Walk Approaches

A recommendation system based on **random walk** uses a random walker that moves in the graph according to a probability of transition.

The probability of reaching a node after a number of steps is used to build a candidate ranking.

These approaches are effective in capturing indirect and superior relationships, especially in heterogeneous graphs.
A common variant is the **random walk with restart**, where the walker can periodically return to the initial node.

Main limits:

- high computational cost;
- heuristic nature, with less optimization than neural models.

### Graph Embedding-Based Recommendation Systems

These approaches map graph nodes in low-dimensional embedding that capture structure and relationships between entities.

Depending on the method used, there are three large families:

- **Graph Factorization-based RS**
- **Graph Distributed Representation-based RS**
- **Graph Neural Embedding-based RS**

#### Graph Factorization-Based RS

They provide co-occurrence matrices or structures derived from meta-path to obtain embedding of users and items.

#### Graph Distributed Representation-Based RS

They learn embeddings through type approaches **Skip-gram** and sequences generated via random walk.

#### Graph Neural Embedding-Based RS

They use neural models, such as MLP or autoencoder, to learn dense representations that can be integrated in end-to-end recommendation systems.

### Graph Neural Network-Based Recommendation Systems

The **GNN-based recommendation systems** directly apply Graph Neural Networks to interaction data.

The main categories are:

- **Graph Attention Network-based RS (GATRS)**
- **Gated Graph Neural Network-based RS (GGNNRS)**
- **Graph Convolutional Network-based RS (GCNRS)**

#### GAT-Based Recommendation Systems

They use mechanisms of attention to weigh differently the influence of nodes and close relationships.

#### GGNN-Based Recommendation Systems

They use gated recurring units to shape complex dynamics and dependencies in sequential graphs.

#### GCN-Based Recommendation Systems

They use convolution on the graph to aggregate structural information and features of the nodes, resulting very effective in the recommendation.

