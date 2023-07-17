import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib inline
sns.set(color_codes=True)

df=pd.read_csv('Dataset.csv')


#VRI
corr_df=df[['temperature','pressure','VRI01','VRI02','VRI03']].corr()

np.tril(np.ones(corr_df.shape)).astype(np.bool)[0:5,0:5]

df_lt = corr_df.where(np.tril(np.ones(corr_df.shape)).astype(np.bool))

df_lt.iloc[0:3,0:5]

hmap=sns.heatmap(df_lt,annot=True, vmax=1, vmin=-1, cmap="Reds")
plt.rcParams["figure.figsize"] = [5, 3]
plt.show()

#VRS
corr_df=df[['temperature','pressure','VRS01', 'VRS03', 'VRS21', 'VRS31','VRS11', 'VRS12','VRS04','VRS22', 'VRS02', 'VRS41', 'VRS32',	'VRS42']].corr()

df_lt2 = corr_df.where(np.tril(np.ones(corr_df.shape)).astype(np.bool))

df_lt2.iloc[0:12,0:19]

hmap=sns.heatmap(df_lt2,annot=True, vmax=1, vmin=-1, cmap=sns.cubehelix_palette(as_cmap=True), annot_kws={"size":9})
plt.rcParams["figure.figsize"] = [16, 5]
plt.show()

#VRR
corr_df=df[['temperature','pressure','VRR12','VRR22','VRR23','VRR33','VRR43','VRR21','VRR34','VRR11','VRR47','VRR32','VRR41','VRR44','VRR42','VRR31','VRR24','VRR13','VRR17','VRR27','VRR14']].corr()

df_lt3.iloc[0:19,0:12]

hmap=sns.heatmap(df_lt3,vmax=1, vmin=-1, annot=True, cbar=True,annot_kws={"size":9}, cbar_kws={'label': 'Relative Displacement Sensors'})
plt.rcParams["figure.figsize"] = [20, 10]
plt.show()
