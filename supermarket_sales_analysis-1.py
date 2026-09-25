"""
Supermarket Sales Analysis
Data Analytics Internship Project

Author: [Your Name]
Description:
    This script loads supermarket sales data from a CSV file, cleans the data,
    calculates sales, performs descriptive analysis, generates charts, and
    exports summary results for internship submission.

Expected CSV columns (case-insensitive):
    Product, Branch, City, Customer type, Quantity, Unit price,
    Payment, Rating

Optional:
    Sales (if already available in the dataset)
"""

from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Configuration
# -----------------------------
INPUT_FILE = "supermarket_sales.csv"
OUTPUT_DIR = Path("supermarket_analysis_output")
OUTPUT_DIR.mkdir(exist_ok=True)


# -----------------------------
# Helper functions
# -----------------------------
def normalize_column_name(column_name: str) -> str:
    """Normalize column names for easier matching."""
    column_name = str(column_name).strip().lower()
    column_name = re.sub(r"[^a-z0-9]+", "_", column_name)
    return column_name.strip("_")


def find_column(df: pd.DataFrame, possible_names: list[str]) -> str | None:
    """Find a column using a list of possible normalized names."""
    normalized = {
        normalize_column_name(column): column for column in df.columns
    }

    for name in possible_names:
        normalized_name = normalize_column_name(name)
        if normalized_name in normalized:
            return normalized[normalized_name]

    return None


def save_bar_chart(series: pd.Series, title: str, xlabel: str, ylabel: str,
                   filename: str, rotate_labels: bool = False) -> None:
    """Create and save a bar chart."""
    plt.figure(figsize=(10, 6))
    series.plot(kind="bar")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if rotate_labels:
        plt.xticks(rotation=45, ha="right")
    else:
        plt.xticks(rotation=0)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close()


# -----------------------------
# 1. Load dataset
# -----------------------------
def load_data(file_path: str) -> pd.DataFrame:
    """Load the CSV dataset."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}\n"
            "Place your CSV file in the same folder as this script "
            "or update INPUT_FILE."
        )

    df = pd.read_csv(path)
    print("\nDataset loaded successfully.")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    return df


# -----------------------------
# 2. Clean and prepare data
# -----------------------------
def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean columns and calculate sales where possible."""
    df = df.copy()
    df.columns = [normalize_column_name(column) for column in df.columns]

    # Convert likely numeric columns
    numeric_candidates = [
        "quantity", "unit_price", "price", "rating", "sales", "total",
        "total_sales"
    ]

    for column in numeric_candidates:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    # Standardize common column names
    if "price" in df.columns and "unit_price" not in df.columns:
        df.rename(columns={"price": "unit_price"}, inplace=True)

    if "customer_type" not in df.columns and "customer" in df.columns:
        df.rename(columns={"customer": "customer_type"}, inplace=True)

    if "payment_method" not in df.columns and "payment" in df.columns:
        df.rename(columns={"payment": "payment_method"}, inplace=True)

    # Calculate sales if sales is not already present
    if "sales" not in df.columns:
        if "quantity" in df.columns and "unit_price" in df.columns:
            df["sales"] = df["quantity"] * df["unit_price"]
        elif "total" in df.columns:
            df["sales"] = df["total"]
        elif "total_sales" in df.columns:
            df["sales"] = df["total_sales"]
        else:
            raise ValueError(
                "Unable to calculate sales. The dataset must contain "
                "quantity and unit price, or an existing sales column."
            )

    # Remove rows with missing essential values
    essential_columns = [
        column for column in
        ["product", "branch", "quantity", "unit_price", "sales"]
        if column in df.columns
    ]

    before = len(df)
    df.dropna(subset=essential_columns, inplace=True)
    after = len(df)

    # Remove invalid numeric values
    for column in ["quantity", "unit_price", "sales", "rating"]:
        if column in df.columns:
            df = df[df[column] >= 0]

    print("\nData cleaning completed.")
    print(f"Rows removed due to missing/invalid values: {before - len(df)}")
    print(f"Rows available for analysis: {len(df)}")

    return df


# -----------------------------
# 3. Analysis
# -----------------------------
def perform_analysis(df: pd.DataFrame) -> dict:
    """Perform the main sales analysis."""
    results = {}

    results["total_sales"] = df["sales"].sum()
    results["total_transactions"] = len(df)

    if "product" in df.columns:
        results["sales_by_product"] = (
            df.groupby("product")["sales"]
            .sum()
            .sort_values(ascending=False)
        )
        results["top_product"] = results["sales_by_product"].idxmax()

    if "branch" in df.columns:
        results["sales_by_branch"] = (
            df.groupby("branch")["sales"]
            .sum()
            .sort_values(ascending=False)
        )
        results["top_branch"] = results["sales_by_branch"].idxmax()

    if "category" in df.columns:
        results["sales_by_category"] = (
            df.groupby("category")["sales"]
            .sum()
            .sort_values(ascending=False)
        )
        results["top_category"] = results["sales_by_category"].idxmax()

    if "payment_method" in df.columns:
        results["payment_counts"] = df["payment_method"].value_counts()
        results["popular_payment"] = results["payment_counts"].idxmax()

    if "customer_type" in df.columns:
        results["average_sales_by_customer"] = (
            df.groupby("customer_type")["sales"].mean().sort_values(ascending=False)
        )

    if "rating" in df.columns:
        results["average_rating"] = df["rating"].mean()

    return results


def print_results(results: dict) -> None:
    """Print analysis results in a readable format."""
    print("\n" + "=" * 60)
    print("SUPERMARKET SALES ANALYSIS RESULTS")
    print("=" * 60)

    print(f"\nTotal sales: ₹{results['total_sales']:,.2f}")
    print(f"Total transactions: {results['total_transactions']}")

    if "sales_by_product" in results:
        print("\nSales by product:")
        print(results["sales_by_product"].round(2))
        print(f"Highest-selling product: {results['top_product']}")

    if "sales_by_branch" in results:
        print("\nSales by branch:")
        print(results["sales_by_branch"].round(2))
        print(f"Best-performing branch: {results['top_branch']}")

    if "sales_by_category" in results:
        print("\nSales by category:")
        print(results["sales_by_category"].round(2))
        print(f"Highest-selling category: {results['top_category']}")

    if "payment_counts" in results:
        print("\nPayment method usage:")
        print(results["payment_counts"])
        print(f"Most popular payment method: {results['popular_payment']}")

    if "average_sales_by_customer" in results:
        print("\nAverage transaction value by customer type:")
        print(results["average_sales_by_customer"].round(2))

    if "average_rating" in results:
        print(f"\nAverage customer rating: {results['average_rating']:.2f} / 5")


# -----------------------------
# 4. Export summaries and charts
# -----------------------------
def export_results(results: dict) -> None:
    """Export summary tables to CSV files."""
    export_map = {
        "sales_by_product": "sales_by_product.csv",
        "sales_by_branch": "sales_by_branch.csv",
        "sales_by_category": "sales_by_category.csv",
        "payment_counts": "payment_method_counts.csv",
        "average_sales_by_customer": "average_sales_by_customer.csv",
    }

    for result_key, filename in export_map.items():
        if result_key in results:
            results[result_key].to_csv(OUTPUT_DIR / filename, header=["value"])

    summary = {
        "total_sales": results.get("total_sales"),
        "total_transactions": results.get("total_transactions"),
        "top_product": results.get("top_product"),
        "top_branch": results.get("top_branch"),
        "top_category": results.get("top_category"),
        "popular_payment": results.get("popular_payment"),
        "average_rating": results.get("average_rating"),
    }

    pd.DataFrame([summary]).to_csv(
        OUTPUT_DIR / "overall_summary.csv", index=False
    )

    print(f"\nSummary files exported to: {OUTPUT_DIR.resolve()}")


def create_charts(results: dict) -> None:
    """Create charts for the analysis."""
    if "sales_by_product" in results:
        save_bar_chart(
            results["sales_by_product"],
            "Sales by Product",
            "Product",
            "Sales",
            "sales_by_product.png",
            rotate_labels=True,
        )

    if "sales_by_branch" in results:
        save_bar_chart(
            results["sales_by_branch"],
            "Sales by Branch",
            "Branch",
            "Sales",
            "sales_by_branch.png",
        )

    if "sales_by_category" in results:
        save_bar_chart(
            results["sales_by_category"],
            "Sales by Category",
            "Category",
            "Sales",
            "sales_by_category.png",
            rotate_labels=True,
        )

    if "payment_counts" in results:
        save_bar_chart(
            results["payment_counts"],
            "Payment Method Usage",
            "Payment Method",
            "Number of Transactions",
            "payment_method_usage.png",
        )

    if "average_sales_by_customer" in results:
        save_bar_chart(
            results["average_sales_by_customer"],
            "Average Transaction Value by Customer Type",
            "Customer Type",
            "Average Sales",
            "average_sales_by_customer.png",
        )

    print("Charts generated successfully.")


# -----------------------------
# 5. Main program
# -----------------------------
def main() -> None:
    """Run the complete project."""
    try:
        data = load_data(INPUT_FILE)
        data = prepare_data(data)

        # Save cleaned data
        data.to_csv(OUTPUT_DIR / "cleaned_supermarket_sales.csv", index=False)

        analysis_results = perform_analysis(data)
        print_results(analysis_results)
        export_results(analysis_results)
        create_charts(analysis_results)

        print("\nProject completed successfully.")

    except Exception as error:
        print(f"\nError: {error}")


if __name__ == "__main__":
    main()
