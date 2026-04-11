# Data libraries
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# Plotting libraries
import matplotlib.pyplot as plt

# Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import ConfusionMatrixDisplay

DATA_FOLDER: str = "./data/"
UNICEF_DATA_CSV: str = DATA_FOLDER + "unicef_malawi.csv"

malawi = pd.read_csv(UNICEF_DATA_CSV)

depression_map: dict[str: int] = {
    "NEVER": 0,
    "A FEW TIMES A YEAR": 0,
    "MONTHLY": 1,
    "WEEKLY": 1,
    "DAILY": 1,
}
malawi["FCF26_binary"] = malawi["FCF26"].map(depression_map)

malawi = malawi[malawi["FCF26_binary"].notna()]

mask = ['HH1',
 'CL3',
 'LS2',
 'LS1',
 'WS4',
 'LS3',
 'HH7',
 'VT22B',
 'MA2',
 'HC4',
 'VT22A',
 'CL13',
 'LS4',
 'WS11',
 'VT22C',
 'HH2',
 'WS7',
 'HC12',
 'MSTATUS',
 'FCD2H',
 'WS1',
 'FCD2J',
 'VT20',
 'HC5',
 'HC8',
 'VT22D',
 'WB4']

malawi = malawi[mask + ["FCF26_binary"]].copy()
malawi = malawi.applymap(lambda x: x.strip().upper() if isinstance(x, str) else x)

malawi = malawi.replace({
    r"NO RESPONSE.*": pd.NA,
    r"^DK.*": pd.NA,
}, regex=True)

# malawi["CB5B"] = (malawi["CB5B"].str.extract(r'(\d+)').astype(float))
# malawi["WB6B"] = (malawi["WB6B"].str.extract(r'(\d+)').astype(float))
# depression_map maps values in FCF26 to binary; where 0 = not depressed, and 1 = depressed.

school_lvl_map: dict[str: int] = {
	"ECE": 1,
	"PRIMARY": 1,
	"LOWER SECONDARY": 2,
	"UPPER SECONDARY": 3,
	"HIGHER": 4,
	"VOCATIONAL TRAINING": 4
}

safety_map: dict[str: int] = {
	"NEVER WALK ALONE AFTER DARK": 0,
	"VERY UNSAFE": 1,
	"UNSAFE": 2,
	"SAFE": 3,
	"VERY SAFE": 4

}

disability_map = {
	"HAS NO FUNCTIONAL DIFFICULTY": 0,
	"HAS FUNCTIONAL DIFFICULTY": 1
}

difficulty_map = {
	"NO DIFFICULTY": 0,
	"SOME DIFFICULTY": 1,
	"A LOT OF DIFFICULTY": 2
}

happiness_map = {
	"VERY UNHAPPY": 0,
	"SOMEWHAT UNHAPPY": 1,
	"NEITHER HAPPY NOR UNHAPPY": 2,
	"SOMEWHAT HAPPY": 3,
	"VERY HAPPY": 4
}

sufficiency_map = {
    "YES, AT LEAST ONCE": 1,
    "NO, ALWAYS SUFFICIENT": 0
}

improvement_map = {
	"WORSENED": -1,
	"WORSE": -1,
	"MORE OR LESS THE SAME": 0,
	"BETTER": 1,
	"IMPROVED": 1
}

facility_location_map = {
	"ELSEWHERE": 0,
	"IN OWN YARD / PLOT": 1,
	"IN OWN DWELLING": 2
}
literacy_map = {
	"NO SENTENCE IN REQUIRED LANGUAGE / BRAILLE": pd.NA,
	"CANNOT READ AT ALL": 0,
	"ABLE TO READ ONLY PARTS OF SENTENCE": 1,
	"ABLE TO READ WHOLE SENTENCE": 2
}
yn_map = {
    "YES": 1,
    "NO": 0
}

# malawi["WB6A"] = malawi["WB6A"].replace(school_lvl_map)
# malawi["CB5A"] = malawi["CB5A"].replace(school_lvl_map)
malawi["WS7"] = malawi["WS7"].replace(sufficiency_map)
# malawi["WS14"] = malawi["WS14"].replace(facility_location_map)
# malawi["WS3"] = malawi["WS3"].replace(facility_location_map)
malawi["LS1"] = malawi["LS1"].replace(happiness_map)
malawi["LS3"] = malawi["LS3"].replace(improvement_map)
malawi["LS4"] = malawi["LS4"].replace(improvement_map)
# malawi["AF10"] = malawi["AF10"].replace(difficulty_map)
# malawi["AF11"] = malawi["AF11"].replace(difficulty_map)
# malawi["AF12"] = malawi["AF12"].replace(difficulty_map)
# malawi["disability"] = malawi["disability"].replace(disability_map)
# malawi["WB14"] = malawi["WB14"].replace(literacy_map)
malawi["VT20"] = malawi["VT20"].replace(safety_map)
# malawi["VT21"] = malawi["VT21"].replace(safety_map)
malawi = malawi.replace(yn_map)

categorical_cols = ["HH7", "MSTATUS", "HC4", "HC5", "HC8", "WS1", "WS11"]
malawi = pd.get_dummies(malawi, columns=categorical_cols, dummy_na=True)

malawi["ws4_na"] = malawi["WS4"].str.contains("MEMBERS DO NOT COLLECT", na=False).astype(int)
malawi["WS4"] = pd.to_numeric(malawi["WS4"], errors="coerce")

for col in malawi.columns:
    if malawi[col].isna().any():
        malawi[col + "_missing"] = malawi[col].isna().astype(int)
        malawi[col] = malawi[col].fillna(-999)

print(malawi.select_dtypes(include="object").columns)

# Feature matrix and response vector
X, y = malawi.drop(["FCF26_binary"], axis=1), malawi["FCF26_binary"]

# Stratify split
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle= True, test_size = 0.2, random_state=42)

# Cost complexity pruning
tree_clf = DecisionTreeClassifier(class_weight={0:1, 1:7.86}, max_depth=10)
tree_clf.fit(X_train, y_train)
path = tree_clf.cost_complexity_pruning_path(X_train,y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

fig, ax = plt.subplots(figsize=(6,5))
ax.plot(ccp_alphas, impurities, marker="o", drawstyle="steps-post")
ax.set_xlabel("alpha")
ax.set_ylabel("total impurity of leaves")
ax.set_title("Total Impurity vs alpha on the training set")
plt.show()

#Code for your answer here
clf = DecisionTreeClassifier(ccp_alpha= 0.002, class_weight={0:1, 1:7.86}, max_depth = 10, criterion="entropy")
clf.fit(X_train, y_train)

#Plot the decision tree

# tree_data = export_graphviz(
#  clf,
#  rounded = True, filled = True, feature_names=X_train.columns, class_names = clf.classes_
# )

plt.figure(figsize=(12,8))
plot_tree(clf, rounded = True, filled = True, feature_names=X_train.columns, class_names = [str(c) for c in clf.classes_])
plt.show()

y_pred = clf.predict(X_test)

ConfusionMatrixDisplay.from_predictions(y_true=y_test, y_pred=y_pred)
plt.show()