from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

filename = "../../data/raw/laptop-data.csv"
df = pd.read_csv(filename)

df.shape

df.info()

# ----------------------------------------------------------------
# Checking for duplicate values
# ----------------------------------------------------------------

df.duplicated().sum()

# ----------------------------------------------------------------
# Cleaning the data set and making it ready for evaluation
# ----------------------------------------------------------------

df.drop(columns="Unnamed: 0", inplace=True)

df["Ram"] = df["Ram"].str.replace("GB", "")
df["Ram"] = df["Ram"].astype("int32")

df["Weight"] = df["Weight"].str.replace("kg", "")
df["Weight"] = df["Weight"].astype("float32")

df.head()


# ----------------------------------------------------------------
# As we can see the data is skewed, this could cause some problems
# ----------------------------------------------------------------

sns.distplot(df["Price"])

# ----------------------------------------------------------------
# Checking number of laptops from various companies
# ----------------------------------------------------------------
df["Company"].value_counts().plot(kind="bar")


# ----------------------------------------------------------------
# This graph shows us which laptops have the highest average cost
# With the company, price varies
# ----------------------------------------------------------------

sns.barplot(x=df["Company"], y=df["Price"])
plt.xticks(rotation="vertical")
plt.show()


# ----------------------------------------------------------------
# Checking the various types of laptops
# ----------------------------------------------------------------

df["TypeName"].value_counts().plot(kind="bar")

sns.barplot(x=df["TypeName"], y=df["Price"])
plt.xticks(rotation="vertical")
plt.show()

# ----------------------------------------------------------------
# Most laptops are in the 15.6 inch category
# We can see that the price increases with the size
# ----------------------------------------------------------------

sns.displot(df["Inches"])
sns.scatterplot(x=df["Inches"], y=df["Price"])


# ----------------------------------------------------------------
# The screen specifications have been given in various different formats
# This type of formatting is extremely unreliable. The information
# obtained from this column is - IPS Display, Resolution, Touch Screen
# ----------------------------------------------------------------

df["ScreenResolution"].value_counts()

df["Touchscreen"] = df["ScreenResolution"].apply(
    lambda x: 1 if "Touchscreen" in x else 0
)

df["Touchscreen"].value_counts().plot(kind="pie")


# ----------------------------------------------------------------
# Touchscreen laptops are costlier than normal laptops, IPS displays
# also follow a similar trend
# ----------------------------------------------------------------

sns.barplot(x=df["Touchscreen"], y=df["Price"])


df["IPS"] = df["ScreenResolution"].apply(lambda x: 1 if "IPS" in x else 0)

df["IPS"].value_counts().plot(kind="bar")

sns.barplot(x=df["IPS"], y=df["Price"])

tempFrame = df["ScreenResolution"].str.split("x", n=1, expand=True)

df["x_resolution"] = tempFrame[0]
df["y_resolution"] = tempFrame[1]

# ----------------------------------------------------------------
# Removing all words from the x resolution column
# ----------------------------------------------------------------

df["x_resolution"] = (
    df["x_resolution"]
    .replace(",", "")
    .str.findall(r"(\d+\.?\d+)")
    .apply(lambda x: x[0])
)

df["x_resolution"] = df["x_resolution"].astype("int32")
df["y_resolution"] = df["y_resolution"].astype("int32")

df.corr()["Price"]


# ----------------------------------------------------------------
# Finding the ppi of the displays
# ----------------------------------------------------------------

df["PPI"] = ((df["x_resolution"] ** 2) + (df["x_resolution"] ** 2)) ** 0.5 / df[
    "Inches"
].astype("float32")


dropped_cols = ["ScreenResolution", "Inches", "x_resolution", "y_resolution"]
df.drop(columns=dropped_cols, inplace=True)

df.head()


df["Cpu"].value_counts()


# ----------------------------------------------------------------
# The number of processors is way too high, simplifying this ->
# ----------------------------------------------------------------

df["Cpu Name"] = df["Cpu"].apply(lambda x: " ".join(x.split()[:3]))


def get_processor(processor_name: str) -> str:
    if (
        processor_name == "Intel Core i7"
        or processor_name == "Intel Core i5"
        or processor_name == "Intel Core i3"
    ):
        return processor_name
    else:
        if processor_name.split()[0] == "Intel":
            return "Misc Intel processor"
        else:
            return "AMD processor"


df["Cpu Brand"] = df["Cpu Name"].apply(get_processor)


df["Cpu Brand"].value_counts().plot(kind="bar")


# ----------------------------------------------------------------
# Seeing how the cpu brand determines the cost of the processor
# ----------------------------------------------------------------


sns.barplot(x=df["Cpu Brand"], y=df["Price"])
plt.xticks(rotation="vertical")
plt.show()


dropped_cols_cpu = ["Cpu", "Cpu Name"]
df.drop(columns=dropped_cols_cpu, inplace=True)

df.head()


df["Ram"].value_counts().plot(kind="bar")

# ----------------------------------------------------------------
# As Ram increases, price increases, relationship is relatively linear
# ----------------------------------------------------------------

sns.barplot(x=df["Ram"], y=df["Price"])
plt.xticks(rotation="vertical")
plt.show()


df["Memory"].value_counts()

# ----------------------------------------------------------------
# Creating 4 different cols for different types of memory
# ----------------------------------------------------------------


df["Memory"] = df["Memory"].astype(str).replace("\.0", "", regex=True)
df["Memory"] = df["Memory"].str.replace("GB", "")
df["Memory"] = df["Memory"].str.replace("TB", "000")
temp_mem = df["Memory"].str.split("+", n=1, expand=True)

df["first"] = temp_mem[0]
df["first"] = df["first"].str.strip()

df["second"] = temp_mem[1]

df["L1_hdd"] = df["first"].apply(lambda x: 1 if "HDD" in x else 0)
df["L1_ssd"] = df["first"].apply(lambda x: 1 if "SSD" in x else 0)
df["L1_hyb"] = df["first"].apply(lambda x: 1 if "Hybrid" in x else 0)
df["L1_fs"] = df["first"].apply(lambda x: 1 if "Flash Storage" in x else 0)

df["first"] = df["first"].str.replace(r"\D", "")

df["second"].fillna("0", inplace=True)

df["L2_hdd"] = df["second"].apply(lambda x: 1 if "HDD" in x else 0)
df["L2_ssd"] = df["second"].apply(lambda x: 1 if "SSD" in x else 0)
df["L2_hyb"] = df["second"].apply(lambda x: 1 if "Hybrid" in x else 0)
df["L2_fs"] = df["second"].apply(lambda x: 1 if "Flash Storage" in x else 0)

df["second"] = df["second"].str.replace(r"\D", "")

df["first"] = df["first"].astype(int)
df["second"] = df["second"].astype(int)

df["HDD"] = df["first"] * df["L1_hdd"] + df["second"] * df["L2_hdd"]
df["SSD"] = df["first"] * df["L1_ssd"] + df["second"] * df["L2_ssd"]
df["Hybrid"] = df["first"] * df["L1_hyb"] + df["second"] * df["L2_hyb"]
df["Flash_Storage"] = df["first"] * df["L1_fs"] + df["second"] * df["L2_fs"]

df.drop(
    columns=[
        "first",
        "second",
        "L1_hdd",
        "L1_ssd",
        "L1_hyb",
        "L1_fs",
        "L2_hdd",
        "L2_ssd",
        "L2_hyb",
        "L2_fs",
    ],
    inplace=True,
)

df.drop(columns=["Memory"], inplace=True)

df.corr()["Price"]

# ----------------------------------------------------------------
# Hybrid and Flash Storage have the least correlation
# ----------------------------------------------------------------

df.drop(columns=["Hybrid", "Flash_Storage"], inplace=True)

# ----------------------------------------------------------------
# Now working on the GPU Columns
# ----------------------------------------------------------------

df["Gpu"].value_counts()

df["Gpu Brand"] = df["Gpu"].apply(lambda x: x.split()[0])

df["Gpu Brand"].value_counts()

df = df[df["Gpu Brand"] != "ARM"]

df.head()


sns.barplot(x=df["Gpu Brand"], y=df["Price"], estimator=np.median)
plt.xticks(rotation="vertical")
plt.show()


# ----------------------------------------------------------------
# Nvidia is the most expensive, AMD however is cheaper than Intel
# ----------------------------------------------------------------

df.drop(columns=["Gpu"], inplace=True)

df.head()

# ----------------------------------------------------------------
# Checking the operating system now
# ----------------------------------------------------------------

df["OpSys"].value_counts()

sns.barplot(x=df["OpSys"], y=df["Price"], estimator=np.median)
plt.xticks(rotation="vertical")
plt.show()


def simplify_os(os_name: str) -> str:
    if os_name == "Windows 10" or os_name == "Windows 7" or os_name == "Windows 10 S ":
        return "Windows"
    elif os_name == "macOS" or os_name == "Mac OS X":
        return "MacOS"
    elif os_name == "Linux":
        return os_name
    else:
        return "No OS / Chrome OS / Android"


df["os"] = df["OpSys"].apply(simplify_os)

df.head()

df.drop(columns=["OpSys"], inplace=True)

df.head()

sns.barplot(x=df["os"], y=df["Price"])
plt.xticks(rotation="vertical")
plt.show()

# ----------------------------------------------------------------
# Checking the weight column
# ----------------------------------------------------------------

sns.displot(df["Weight"])
sns.scatterplot(x=df["Weight"], y=df["Price"])

# ----------------------------------------------------------------
# There is a very weak linear relationship between weight and price
# ----------------------------------------------------------------

sns.heatmap(df.corr())

sns.displot(np.log(df["Price"]))

# ----------------------------------------------------------------
# Exporting the final csv dataframe
# ----------------------------------------------------------------


output_file = "../../data/processed/final-df.csv"
df.to_csv(output_file)
