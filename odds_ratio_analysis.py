# Meningioma ML Project - Sanchit Aapan

# Import relevant libraries
import pandas as pd
from scipy.stats import fisher_exact
from plotly import colors
import plotly.graph_objects as go


# Import data into a Pandas dataframe
data = pd.read_csv('meningioma_full_data.csv')


# Split data into input features (X) and target variable (y)
# Drop numerical features (NDS score, PMHX score, Age)
X = data.drop(['MRN', 'Comp', 'NDS score', 'PMHX score', 'Age'], axis=1)
y = data['Comp']


features = [i for i in X.columns]

sig = []
odds = []
ps = []

alpha = 0.1

# Loop through each binary feature and perform Fisher's exact test
for feature in features:
    # Create a 2x2 contingency table
    contingency_table = pd.crosstab(data[feature], data['Comp'])
    print(contingency_table)

    # Perform Fisher's exact test
    odds_ratio, p_value = fisher_exact(contingency_table)

    if p_value < alpha:
        odds.append(odds_ratio)
        ps.append(p_value)
        sig.append((feature, odds_ratio))
    else:
        odds.append(0)
        ps.append(0.15)

    # Print the results
    print(f"Fisher's exact test for {feature}:")
    print(f"Odds Ratio: {odds_ratio}")
    print(f"P-value: {p_value}\n")


# Create a heatmap with specified values, colour scale and layout for odds ratios

custom_colorscale = colors.sequential.Blues

heatmap = go.Heatmap(z=[odds[0:5], odds[5:10], odds[10:15], odds[15:20], odds[20:25], odds[25:30], odds[30:35], odds[35:40], odds[40:45]],
                    text=[features[0:5], features[5:10], features[10:15], features[15:20], features[20:25], features[25:30], features[30:35], features[35:40], features[40:45]],
                    texttemplate="%{text}",
                    textfont={"size":20},
                    colorscale=custom_colorscale)

layout = go.Layout(
    xaxis=dict(
        showticklabels=False,
        ticks='',
        showgrid=False,
    ),
    yaxis=dict(
        showticklabels=False,
        ticks='',
        showgrid=False
    )
)

# Create and show the figure with the specified heatmap and layout
fig = go.Figure(data=[heatmap], layout=layout)
fig.show()


# Create a heatmap with specified values, colour scale and layout for p-values

custom_colorscale_1 = colors.sequential.Reds

heatmap_1 = go.Heatmap(z=[ps[0:5], ps[5:10], ps[10:15], ps[15:20], ps[20:25], ps[25:30], ps[30:35], ps[35:40], ps[40:45]],
                    text=[features[0:5], features[5:10], features[10:15], features[15:20], features[20:25], features[25:30], features[30:35], features[35:40], features[40:45]],
                    texttemplate="%{text}",
                    textfont={"size":20},
                    colorscale=custom_colorscale_1)

layout_1 = go.Layout(
    xaxis=dict(
        showticklabels=False,
        ticks='',
        showgrid=False
    ),
    yaxis=dict(
        showticklabels=False,
        ticks='',
        showgrid=False
    )
)

# Create and show the figure with the specified heatmap and layout
fig_1 = go.Figure(data=[heatmap_1], layout=layout_1)
fig_1.show()
