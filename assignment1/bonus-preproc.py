import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier as DTC

# load the original data into a DataFrame
data = np.loadtxt('data/encoded_data.csv', delimiter=' ')
df = pd.DataFrame(data,columns=['issuercountry', 'txvariantcode', 'issuer_id', 'amount', 'currencycode', 'shoppercountry', 'interaction', 'verification', 'cvcresponse', 'creationdate_stamp', 'accountcode', 'mail_id', 'ip_id', 'card_id', 'label'])

# compute mean of features for groups of transactions with the
# same creation date timestamp, card id, mail id, and ip id
gdf = df.groupby(['creationdate_stamp']).agg('mean')
gdf2 = df.groupby(['card_id']).agg('mean')
gdf3 = df.groupby(['mail_id']).agg('mean')
gdf4 = df.groupby(['ip_id']).agg('mean')

# fix DataFrames so that the grouping feature becomes a column again
gdf = gdf.reset_index()
gdf2 = gdf2.reset_index()
gdf3 = gdf3.reset_index()
gdf4 = gdf4.reset_index()


# iterate initial transaction list and append the mean amount
# spent for that particular ip, email, card and time
agg = []
for i in range(df.shape[0]):
    result4 = gdf4.loc[gdf4['ip_id'] == df.loc[i]['ip_id'], 'amount']
    result3 = gdf3.loc[gdf3['mail_id'] == df.loc[i]['mail_id'], 'amount']
    result2 = gdf2.loc[gdf2['card_id'] == df.loc[i]['card_id'], 'amount']
    result = gdf.loc[gdf['creationdate_stamp'] == df.loc[i]['creationdate_stamp'], 'amount']
    agg.append(list(df.loc[i]) + list(result) + list(result2) + list(result3) + list(result4))

# create the augmented dataset DataFrame
df2 = pd.DataFrame(agg, columns=['issuercountry', 'txvariantcode', 'issuer_id', 'amount', 'currencycode', 'shoppercountry', 'interaction', 'verification', 'cvcresponse', 'creationdate_stamp', 'accountcode', 'mail_id', 'ip_id', 'card_id', 'label', 'mean_amount_per_date', 'mean_amount_per_car', 'mean_amount_per_mail', 'mean_amount_per_ip'])

# save dataset to a new .csv file
df2.to_csv('data/augmented_data.csv')
