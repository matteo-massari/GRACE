class DataClass:
    """
    Store the canonical column names and graph metadata used across the dataset
    preparation and recommendation pipeline.

    This class centralizes:
    - the names of the user, item, and timestamp columns
    - the names of the graph edge types
    - the product feature schema
    - the customer feature schema

    It is used as a lightweight shared configuration object so that all
    components in the pipeline refer to the same dataset conventions.

    Attributes:
        user_name (str):
            Name of the user identifier column.
        item_name (str):
            Name of the item/product identifier column.
        time_col (str):
            Name of the timestamp column.

        user_to_product (str):
            Name of the forward edge relation from users to products.
        product_to_user (str):
            Name of the reverse edge relation from products to users.

        product_categorical_cols (list[str]):
            List of categorical product feature columns.
        product_numerical_cols (list[str]):
            List of numerical product feature columns.
        product_feature_cols (list[str]):
            Full list of product feature columns.

        customer_categorical_cols (list[str]):
            List of categorical customer feature columns.
        customer_numerical_cols (list[str]):
            List of numerical customer feature columns.
        customer_feature_cols (list[str]):
            Full list of customer feature columns.
    """

    def __init__(self) -> None:
        """
        Initialize the dataset schema configuration.
        """
        self.user_name: str = "ids_customer"
        self.item_name: str = "ids_product"
        self.time_col: str = "dat"

        # Graph edge relation names
        self.user_to_product: str = "buys"
        self.product_to_user: str = "rev_buys"

        # Product features
        self.product_categorical_cols: list[str] = []
        self.product_numerical_cols: list[str] = []
        self.product_feature_cols: list[str] = (
            self.product_categorical_cols + self.product_numerical_cols
        )

        # Customer features
        self.customer_categorical_cols: list[str] = [
            "ids_customer",
            "sector",
            "potential",
            "county",
            "creditlimit",
        ]
        self.customer_numerical_cols: list[str] = []
        self.customer_feature_cols: list[str] = (
            self.customer_categorical_cols + self.customer_numerical_cols
        )
