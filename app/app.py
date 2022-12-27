import numpy as np
import streamlit as st
import pickle

pipe = pickle.load(open("../models/pipe.pkl", "rb"))
df = pickle.load(open("../models/df.pkl", "rb"))

st.title("Laptop Price Prediction")

brand_selection = st.selectbox("Select Brand", df["Company"].unique())
type_selection = st.selectbox("Select Type Of Laptop", df["TypeName"].unique())
ram_selection = st.selectbox(
    "Select Amount of Ram in Gb", [2, 4, 6, 8, 12, 16, 24, 32, 64]
)

weight = st.number_input("Weight")

touchscreen = st.selectbox("Touch Screen(Y/N)", ["Yes", "No"])
ips = st.selectbox("IPS Display(Y/N)", ["Yes", "No"])

screen_size = st.number_input("Screen size")

resolutions = [
    "1920x1080",
    "1366x768",
    "1600x900",
    "3840x2160",
    "3200x1800",
    "2880x1800",
    "2560x1600",
    "2560x1440",
    "2304x1440",
]
resolution = st.selectbox("Resolution", resolutions)

cpu_selection = st.selectbox("CPU Brand", df["Cpu Brand"].unique())

hdd_selection = st.selectbox("HDD storage(in GB)", [0, 128, 256, 512, 1024, 2048])

ssd_selection = st.selectbox("SSD storage(in GB)", [0, 8, 128, 256, 512, 1024])

gpu_selection = st.selectbox("GPU brand", df["Gpu Brand"].unique())

os_selection = st.selectbox("Operating System", df["os"].unique())


if st.button("Predict Price"):
    ppi = None
    if touchscreen == "Yes":
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == "Yes":
        ips = 1
    else:
        ips = 0

    X_res = int(resolution.split("x")[0])
    Y_res = int(resolution.split("x")[1])
    ppi = ((X_res**2) + (Y_res**2)) ** 0.5 / screen_size
    query = np.array(
        [
            brand_selection,
            type,
            ram_selection,
            weight,
            touchscreen,
            ips,
            ppi,
            cpu_selection,
            hdd_selection,
            ssd_selection,
            gpu_selection,
            os_selection,
        ]
    )

    query = query.reshape(1, 12)
    st.title(
        "The predicted price of the laptop is:  "
        + str(int(np.exp(pipe.predict(query)[0])))
    )
