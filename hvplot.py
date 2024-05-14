import pandas as pd
import hvplot.pandas
import holoviews as hv

# load in data and postprocess input
df = (
    pd.read_csv(
        "https://mesonet.agron.iastate.edu/cgi-bin/request/daily.py?network=WA_ASOS&stations=SEA&year1=1928&month1=1&day1=1&year2=2023&month2=12&day2=31&var=max_temp_f&var=min_temp_f&var=precip_in&na=blank&format=csv",
        parse_dates=True,
        index_col="day",
    )
    .drop(columns=["station"])
    .astype("float16")
    .assign(
        dayofyear=lambda df: df.index.dayofyear,
        year=lambda df: df.index.year,
    )
)

# get average and 2023
df_avg = df.loc[df["year"].between(1990, 2020)].groupby("dayofyear").mean()
df_2023 = df[df.year == 2023]

# preprocess below/above
df_above = df_2023[["dayofyear", "max_temp_f"]].merge(
    df_avg.reset_index()[["dayofyear", "max_temp_f"]],
    on="dayofyear",
    suffixes=("_2023", "_avg"),
)
df_above["max_temp_f"] = df_above["max_temp_f_avg"]
df_above["max_temp_f"] = df_above.loc[df_above["max_temp_f_2023"] >= df_above["max_temp_f_avg"], "max_temp_f_2023"]

df_below = df_2023[["dayofyear", "max_temp_f"]].merge(
    df_avg.reset_index()[["dayofyear", "max_temp_f"]],
    on="dayofyear",
    suffixes=("_2023", "_avg"),
)
df_below["max_temp_f"] = df_below["max_temp_f_avg"]
df_below["max_temp_f"] = df_below.loc[df_below["max_temp_f_2023"] < df_below["max_temp_f_avg"], "max_temp_f_2023"]

days_above = df_above.query("max_temp_f_2023 >= max_temp_f_avg")["max_temp_f"].size
days_below = df_below.query("max_temp_f_2023 < max_temp_f_avg")["max_temp_f"].size

# create plot elements
plot = df.hvplot(x="dayofyear", y="max_temp_f", by="year", color="grey", alpha=0.02, legend=False, hover=False)
plot_avg = df_avg.hvplot(x="dayofyear", y="max_temp_f", color="grey", legend=False)
plot_2023 = df_2023.hvplot(x="dayofyear", y="max_temp_f", color="black", legend=False)

dark_red = "#FF5555"
dark_blue = "#5588FF"

plot_above = df_above.hvplot.area(
    x="dayofyear", y="max_temp_f_avg", y2="max_temp_f"
).opts(fill_alpha=0.2, line_alpha=0.8, line_color=dark_red, fill_color=dark_red)
plot_below = df_below.hvplot.area(
    x="dayofyear", y="max_temp_f_avg", y2="max_temp_f"
).opts(fill_alpha=0.2, line_alpha=0.8, line_color=dark_blue, fill_color=dark_blue)

text_days_above = hv.Text(
    35, df_2023["max_temp_f"].max(), f"{days_above}", fontsize=14
).opts(text_align="right", text_baseline="bottom", text_color=dark_red, text_alpha=0.8)
text_days_below = hv.Text(
    35, df_2023["max_temp_f"].max(), f"{days_below}", fontsize=14
).opts(text_align="right", text_baseline="top", text_color=dark_blue, text_alpha=0.8)
text_above = hv.Text(38, df_2023["max_temp_f"].max(), "DAYS ABOVE", fontsize=7).opts(
    text_align="left", text_baseline="bottom", text_color="lightgrey", text_alpha=0.8
)
text_below = hv.Text(38, df_2023["max_temp_f"].max(), "DAYS BELOW", fontsize=7).opts(
    text_align="left", text_baseline="above", text_color="lightgrey", text_alpha=0.8
)

# overlay everything and save
final = (
    plot
    * plot_avg
    * plot_above
    * plot_below
    * text_days_above
    * text_days_below
    * text_above
    * text_below
).opts(
    xlabel="TIME OF YEAR",
    ylabel="MAX TEMP °F",
    title="SEATTLE 2023 vs AVERAGE (1990-2020)",
    gridstyle={"ygrid_line_alpha": 0},
    xticks=[
        (1, "JAN"),
        (31, "FEB"),
        (59, "MAR"),
        (90, "APR"),
        (120, "MAY"),
        (151, "JUN"),
        (181, "JUL"),
        (212, "AUG"),
        (243, "SEP"),
        (273, "OCT"),
        (304, "NOV"),
        (334, "DEC"),
    ],
    show_grid=True,
    fontscale=1.18,
)
hv.save(final, "final.html")