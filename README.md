
# Market Basket Analysis in Python using Apriori Algorithm


from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Sample transaction data
transactions = [['bread', 'milk', 'eggs'],
                ['bread', 'milk'],
                ['milk', 'eggs'],
                ['bread', 'butter'],
                ['bread', 'milk', 'butter'],
                ['bread', 'milk', 'eggs', 'butter'],
                ['milk', 'butter']]

# Convert the transaction data into a one-hot encoded matrix
te = TransactionEncoder()
te_ary = te.fit_transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Apply the Apriori algorithm to find frequent itemsets
frequent_itemsets = apriori(df, min_support=0.2, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# Print the frequent itemsets
print("Frequent Itemsets:")
print(frequent_itemsets)

# Print the association rules
print("Association Rules:")
print(rules)

