# import plotly.express as px
# gapminder = px.data.gapminder().query("year==2007")
# fig = px.scatter_geo(gapminder, locations="iso_alpha", color="continent",
#                      hover_name="country", size="pop",
#                      projection="natural earth")
# fig.show()

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = np.arange(23)
y = np.array([110,82.5,102,84,76,104.5,82.5,62,89,75,
              85.5,94.5,87,99,77,74.5,94,78,95.5,94.5,
              87,108.5,75])

df = pd.DataFrame({"x-axis": x,"y-axis": y})

sns.pointplot("x-axis","y-axis",palette="Paired",data=df)
#plt.xticks(rotation=90)
plt.show()