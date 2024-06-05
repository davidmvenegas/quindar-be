import pandas as pd
import numpy as np


MAXIMUM_FUEL_CAPACITY = 20
LARGE_DATASET_THRESHOLD = 50


def main(filename, algorithm="auto"):
    """
    The main entry point which will select between a greedy or DP algorithm based on data size (or override).

    Parameters:
        filename (str): Path to the CSV file containing the service opportunities.
        algorithm (str): The algorithm to use. Can be 'greedy', 'dp', or 'auto' (default).

    Returns:
        list of dicts: Selected opportunities.
        float: Total revenue from the selected opportunities.
    """

    dataframe = pd.read_csv(filename)
    dataframe["efficiency_ratio"] = dataframe["revenue"] / dataframe["fuel_cost"]

    num_entries = len(dataframe)

    if algorithm == "greedy":
        return greedy_algorithm(dataframe)
    elif algorithm == "dp":
        return dynamic_programming_algorithm(dataframe)
    elif algorithm == "auto":
        if num_entries > LARGE_DATASET_THRESHOLD:
            return dynamic_programming_algorithm(dataframe)
        else:
            return greedy_algorithm(dataframe)
    else:
        raise ValueError("Invalid override option. Choose 'greedy', 'dp', or 'auto'.")


def greedy_algorithm(dataframe):
    print("Running greedy algorithm")

    # Sort the opportunities by efficiency ratio in descending order.
    sorted_opportunities = dataframe.sort_values(by="efficiency_ratio", ascending=False)

    # Initialize variables to keep track of the total revenue and fuel used by selected opportunities.
    total_revenue = 0
    total_fuel_used = 0

    # A list to store the opportunities we decide to take.
    selected_opportunities = []

    # Iterate over each opportunity in the sorted list.
    for _, opportunity in sorted_opportunities.iterrows():
        # Check if adding this opportunity exceeds the maximum fuel capacity.
        if total_fuel_used + opportunity["fuel_cost"] <= MAXIMUM_FUEL_CAPACITY:
            # Update the total fuel used and total revenue with this opportunity's values.
            total_fuel_used += opportunity["fuel_cost"]
            total_revenue += opportunity["revenue"]

            # Add this opportunity to the list of selected ones.
            selected_opportunities.append(opportunity)

    # Return the list of selected opportunities and the total revenue obtained.
    return selected_opportunities, total_revenue


def dynamic_programming_algorithm(dataframe):
    print("Running dynamic programming algorithm")

    # Convert each opportunity into a dictionary and ensure fuel costs are integers.
    opportunities = dataframe.to_dict("records")
    for opp in opportunities:
        opp["fuel_cost"] = int(opp["fuel_cost"])

    # Initialize a table to store maximum revenue at each combination of opportunities and fuel usage.
    num_opportunities = len(opportunities)
    max_fuel = MAXIMUM_FUEL_CAPACITY
    dp = [[0] * (max_fuel + 1) for _ in range(num_opportunities + 1)]

    # Fill the dp table where each cell represents the maximum revenue achievable with a given fuel capacity.
    for i in range(1, num_opportunities + 1):
        for fuel in range(max_fuel + 1):
            current_fuel_cost = opportunities[i - 1]["fuel_cost"]
            current_revenue = opportunities[i - 1]["revenue"]
            if current_fuel_cost <= fuel:
                dp[i][fuel] = max(
                    dp[i - 1][fuel],
                    dp[i - 1][fuel - current_fuel_cost] + current_revenue,
                )
            else:
                dp[i][fuel] = dp[i - 1][fuel]

    # Determine which opportunities to select to achieve the maximum revenue without exceeding the fuel capacity.
    selected_opportunities = []
    remaining_fuel = max_fuel
    for i in range(num_opportunities, 0, -1):
        if (
            remaining_fuel >= opportunities[i - 1]["fuel_cost"]
            and dp[i][remaining_fuel]
            == dp[i - 1][remaining_fuel - opportunities[i - 1]["fuel_cost"]]
            + opportunities[i - 1]["revenue"]
        ):
            selected_opportunities.append(opportunities[i - 1])
            remaining_fuel -= opportunities[i - 1]["fuel_cost"]

    # Total revenue is found at the last cell of the table corresponding to the maximum fuel capacity.
    total_revenue = dp[num_opportunities][max_fuel]
    return selected_opportunities, total_revenue


# Run the main function with the provided CSV file.
# OPTIONAL: Override the algorithm by passing algorithm="greedy" or algorithm="dp".
selected_ops, total_rev = main("servicing_options.csv")

# Output the total revenue and the number of opportunities selected.
print(f"Total Revenue: {total_rev} USD, Total Opportunities: {len(selected_ops)}")
