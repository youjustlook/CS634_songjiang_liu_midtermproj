# import libraries
import pandas as pd
import numpy as np
from apriori_python.apriori import apriori
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder
import itertools
import time

print("Hello, welcome to the Apriori Algorithm. Version 1.0")

def main():
    # load the database at choice
    databases = ["Amazon", "BestBuy", "HomeDepot", "K-mart", "Nike"]
    print("Please select a database to load:")
    for i, db in enumerate(databases, 1):
        print(f"{i}. {db}")
    
    while True:
        try:
            choice = int(input("Enter the number of your choice: "))
            transactions_data = load_database(choice)
            transactions_data_each = transactions_data.iloc[:, 1]
            if transactions_data is not None:
                print(f"Loaded {databases[choice-1]} database successfully. \n--------------------")
                        
                # Display number of transactions
                num_transactions = len(transactions_data)
                print(f"Number of transactions: {num_transactions}")
                
                # Display item set information
                item_set = get_item_set(transactions_data)
                num_items = len(item_set)
                print(f"Number of items in the item set: {num_items} \n--------------------")
                print("Items in the item set:")
                for item in item_set:
                    print(item)
                print("--------------------")

                # Get user input for minimum support and minimum confidence
                while True:
                    try:
                        min_support = float(input("Enter the minimum support level in % (larger than 0 and lower than 100%): "))
                        if 0 < min_support < 100:
                            break
                        else:
                            print("Invalid input. Please enter a value between 0 and 100.")
                    except ValueError:
                        print("Invalid input. Please enter a numeric value.")

                while True:
                    try:
                        min_confidence = float(input("Enter the minimum confidence level in % (larger than 0 and lower than 100%): "))
                        if 0 < min_confidence < 100:
                            break
                        else:
                            print("Invalid input. Please enter a value between 0 and 100.")
                    except ValueError:
                        print("Invalid input. Please enter a numeric value.")
                order = sorted(item_set)
                transactions_data_processed = []
                for lines in transactions_data_each:
                    trans = list(lines.strip().split(', '))
                    trans_l = list(np.unique(trans))
                    trans_l.sort(key=lambda x: order.index(x))
                    transactions_data_processed.append(sorted(trans_l))
                # print("Transactions data processed: " + str(transactions_data_processed)+"\n")

                # Brute force method
                start_time_brute_force = time.time_ns()
                frequent_itemsets = brute_force_apriori(transactions_data_processed, item_set, min_support / 100)
                brute_force_time = time.time_ns() - start_time_brute_force
                print("--------------------\nFrequent item sets using brute force method: item | support")
                for itemset in frequent_itemsets:
                    print(str(itemset) + " | " + str(count_frequent(itemset, transactions_data_processed)/len(transactions_data_processed)))

                print("--------------------\nAssociation rules using brute force method:")
                start_time_self_association_rules = time.time_ns()
                rules = find_association_rules(frequent_itemsets, transactions_data_processed, min_confidence / 100)
                find_rules_time = time.time_ns() - start_time_self_association_rules
                for i, rule in enumerate(rules):
                    antecedent, consequent, support, confidence = rule
                    print(f"Rule {i+1}: {antecedent} -> {consequent} (support: {support}, confidence: {confidence})")

                # check with built-in apriori algorithm
                print("--------------------\nCheck with Built-in Apriori Algorithm:")
                start_time_apriori = time.time_ns()
                freq_item_set, rules = apriori(transactions_data_processed, min_support/100, min_confidence/100)
                apriori_time = time.time_ns() - start_time_apriori
                print("--------------------\nFrequent item sets with Built-in Apriori algorithm:")
                print(freq_item_set)
                
                print("--------------------\nAssociation rules with Built-in Apriori algorithm:")
                for i, rule in enumerate(rules):
                    print(f"Rule {i+1}: {rule}")

                # check with fp-growth algorithm
                # Convert the transactions into a one-hot encoded DataFrame for fpgrowth
                te = TransactionEncoder()
                te_ary = te.fit(transactions_data_processed).transform(transactions_data_processed)
                df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
                start_time_fpgrowth = time.time_ns()
                freq_item_set_fp = fpgrowth(df_encoded, min_support/100, use_colnames=True)
                fpgrowth_time = time.time_ns() - start_time_fpgrowth
                print("--------------------\nCheck with Built-in FP-Growth algorithm:")
                # print(freq_item_set_fp)
                # Step 2: Generate association rules based on confidence
                start_time_association_rules = time.time_ns()
                rules_fp = association_rules(freq_item_set_fp, metric="confidence", min_threshold=min_confidence / 100)
                association_rules_time = time.time_ns() - start_time_association_rules
                print("--------------------\nAssociation rules with FP-Growth algorithm: rule, support, confidence")
                for idx, row in rules_fp.iterrows():
                    antecedent = set(row['antecedents'])
                    consequent = set(row['consequents'])
                    support = row['support']
                    confidence = row['confidence']
                    print(f"Rule {idx + 1}: {antecedent} -> {consequent} (support: {support:.2f}, confidence: {confidence:.2f})")

                # Print combined execution times
                print("\nExecution Times (in nanoseconds):")
                print(f"Self-Built Brute Force: {brute_force_time + find_rules_time} ns")
                print(f"Built-in Apriori: {apriori_time} ns")
                print(f"Built-In FP-Growth: {fpgrowth_time + association_rules_time} ns")

                break
        except ValueError:
            print("Invalid input. Please enter a valid number.")


# Apriori Algorithm Implementation

def load_database(choice):
    databases = {
        1: "Amazon.csv",
        2: "BestBuy.csv",
        3: "HomeDepot.csv",
        4: "K-mart.csv",
        5: "Nike.csv"
    }
    try:
        return pd.read_csv(databases[choice])
    except KeyError:
        print("Invalid choice. Please enter a number between 1 and 5.")
        return None
    except FileNotFoundError:
        print(f"File {databases[choice]} not found.")
        return None

def get_item_set(transactions_data):
    item_set = set()
    for transaction in transactions_data.iloc[:, 1]:
        items = transaction.split(', ')
        item_set.update(items)
    return item_set


# brute force method
def generate_candidates(item_set, length):
    return [set(item) for item in itertools.combinations(item_set, length)]

def is_frequent(candidate, transactions, min_support_count):
    count = sum(1 for transaction in transactions if candidate.issubset(transaction))
    return count >= min_support_count

def count_frequent(candidate, transactions):
    count = sum(1 for transaction in transactions if candidate.issubset(transaction))
    return count

def brute_force_apriori(transactions, item_set, min_support):
    min_support_count = len(transactions) * min_support
    frequent_itemsets = []
    k = 1
    current_itemsets = [set([item]) for item in item_set]

    while current_itemsets:
        next_itemsets = []
        for itemset in current_itemsets:
            if is_frequent(itemset, transactions, min_support_count):
                frequent_itemsets.append(itemset)
        k += 1
        current_itemsets = generate_candidates(set(itertools.chain.from_iterable(frequent_itemsets)), k)

    return frequent_itemsets


def find_association_rules(frequent_collections, data_samples, threshold_confidence):
    association_rules = []
    for collection in frequent_collections:
        for num_elements in range(1, len(collection)):
            for precursor in itertools.combinations(collection, num_elements):
                precursor = set(precursor)
                outcome = collection - precursor
                if outcome:
                    support = sum(1 for sample in data_samples if collection.issubset(sample)) / len(data_samples)
                    confidence = support / (sum(1 for sample in data_samples if precursor.issubset(sample)) / len(data_samples))
                    if confidence >= threshold_confidence:
                        association_rules.append((precursor, outcome, support, confidence))
    return association_rules


if __name__ == "__main__":
    main()
