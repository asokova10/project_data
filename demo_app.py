import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import plotly
import plotly.graph_objs as go
from geopy.geocoders import Nominatim
import folium
from folium.plugins import MarkerCluster
import geopandas as gpd
import plotly.express as px
import pycountry
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from plotly.subplots import make_subplots
from streamlit_folium import folium_static
import requests
from scipy import stats
import sqlite3
from PIL import Image
from bs4 import BeautifulSoup
import networkx as nx
with st.echo(code_location='below'):
    def main():
        df = pd.read_csv('~/Desktop/world-happiness-report.csv')
        df_2021 = pd.read_csv('~/Desktop/world-happiness-report-2021.csv')
        p = st.sidebar.selectbox("Выберите страничку", ["Описание проекта", "Изучение данных", "Изучение данных. Медленная таблица" ,"Планируем переезд",
                                                           "Немножко ML", "Победитель конкурса 'Лучшая страна'"])


        if p == "Описание проекта":
            st.markdown("Онлайн-визуализатор статистики по индексу счастья")
            st.markdown("Выберите страницу слева.")
            st.markdown("На второй страничке вы найдете R и Julia + визуализации на Python. На третьей Folium, четвертой - API REST и SQL. На пятой - машинное обучение + математические возможности. На шестой - веб-скреппинг")
            st.markdown("ВНИМАНИЕ: 'Изучение данных. Медленная карта' реально медленнная. Минуты 2-3. Keep calm")
            st.markdown("Про длину: есть еще код на r и julia. Он отдельно")
            st.markdown("Исходный датасет:")
            st.write(df)

        elif p == "Изучение данных":

            st.markdown("Изучим наши данные!")
            st.markdown("Сперва предлагаю вам посмотреть на распределение Ladder score по странам на карте")
            st.markdown("Для построения этой карты добавим новый столбец ISO в датафрейм")
            df_2021 = pd.read_csv('~/Desktop/world-happiness-report-2021.csv')
            df = pd.read_csv('~/Desktop/world-happiness-report.csv')


            def get_iso(c_name):
                q = pycountry.countries.get(name=c_name)
                if q:
                    return q.alpha_3

            df_2021['iso_alpha'] = list(map(get_iso, df_2021['Country name']))

            fig = px.choropleth(df_2021, locations="iso_alpha",
                            color="Ladder score",
                            hover_name="Country name",
                            color_continuous_scale=px.colors.sequential.Viridis,
                            range_color=(df_2021['Ladder score'].min(), df_2021['Ladder score'].max()),
                            width=900,
                            height=600,
                            labels={'Ladder score': 'Ladder score'})
            st.plotly_chart(fig)
            st.markdown("Чем богаче - тем счасливее?")
            st.markdown("Посмотрим, как Ladder score зависит от ВВП (можно посмотреть тренд по миру и по регионам)")
            scale_type = st.radio("Зависимость коэффицента счастья от ВВП: ", ['по миру', 'по регионам'])
            if scale_type == "по миру":
                fig = px.scatter(df_2021,
                                 x="Logged GDP per capita",
                                 y="Ladder score",
                                 # color="Regional indicator",
                                 hover_name="Country name",
                                 trendline="ols")
                st.plotly_chart(fig)
            else:
                fig = px.scatter(df_2021,
                                 x="Logged GDP per capita",
                                 y="Ladder score",
                                 color="Regional indicator",
                                 hover_name="Country name",
                                 trendline="ols")
                st.plotly_chart(fig)

            st.markdown("Теперь посмотрим, как дела обстоят по регионам")
            st.markdown("Распеределим все страны по реагионам, к которым они принадлежат, и построим график: с одной стороны кол-во стран в регионе, с другой - средний ladder score региона")

            number_of_countries = []
            mean_ladder = []
            for j in list(set(df_2021["Regional indicator"])):
                country_counter = 0
                sum_ladder = 0
                for i in range(len(df_2021)):
                    if df_2021.iloc[i]["Regional indicator"] == j:
                        country_counter += 1
                        sum_ladder += df_2021.iloc[i]["Ladder score"]
                number_of_countries.append(country_counter)
                mean_ladder.append(sum_ladder / country_counter)
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=list(set(df_2021["Regional indicator"])),
                y=number_of_countries,
                name='Number of countries',
                marker_color='lightblue',
                text=number_of_countries,
                textposition='inside',
                yaxis='y1'
            ))
            fig.add_trace(go.Scatter(
                x=list(set(df_2021["Regional indicator"])),
                y=mean_ladder,
                name='Average Ladder Score',
                mode='markers+text+lines',
                marker_color='black',
                marker_size=10,
                text=mean_ladder,
                textposition='top center',
                line=dict(color='blue', dash='dash'),
                yaxis='y2'

            ))
            fig.update_layout(

                xaxis_title="Region",
                yaxis_title="Number of countries",
                plot_bgcolor='white',
                yaxis2=dict(showgrid=True, overlaying='y', side='right', title='Avg. Ladder Score'),
                legend=dict(yanchor="top",
                            y=1.3,
                            xanchor="left",
                            x=0.78)
            )
            st.plotly_chart(fig)
            st.markdown("Ну и небольшую анимацию! Зависимость Ladder score от ВВП.")
            st.markdown("Для этой визуализации совместим две таблицы, используя возможности pandas. Необходимо в таблицу, где есть данные по большому количеству лет, добавить регионы. Так же названия колонок двух таблиц не всегда совпадают, что создало дополнительную трудность")
            country_continent = {}
            for i in range(len(df_2021)):
                country_continent[df_2021['Country name'][i]] = df_2021['Regional indicator'][i]

            region = []
            for i in range(len(df)):
                if df['Country name'][i] == 'Angola':
                    region.append("Sub-Saharan Africa")
                elif df['Country name'][i] == 'Belize':
                    region.append("Latin America and Caribbean")
                elif df['Country name'][i] == 'Congo (Kinshasa)':
                    region.append("Sub-Saharan Africa")
                elif df['Country name'][i] == 'Syria':
                    region.append("Middle East and North Africa")
                elif df['Country name'][i] == 'Trinidad and Tobago':
                    region.append("Latin America and Caribbean")
                elif df['Country name'][i] == 'Cuba':
                    region.append("Latin America and Caribbean")
                elif df['Country name'][i] == 'Qatar':
                    region.append("Middle East and North Africa")
                elif df['Country name'][i] == 'Sudan':
                    region.append("Middle East and North Africa")
                elif df['Country name'][i] == 'Central African Republic':
                    region.append("Sub-Saharan Africa")
                elif df['Country name'][i] == 'Djibouti':
                    region.append("Sub-Saharan Africa")
                elif df['Country name'][i] == 'Somaliland region':
                    region.append("Sub-Saharan Africa")
                elif df['Country name'][i] == 'South Sudan':
                    region.append("Middle East and North Africa")
                elif df['Country name'][i] == 'Somalia':
                    region.append("Sub-Saharan Africa")
                elif df['Country name'][i] == 'Oman':
                    region.append("Middle East and North Africa")
                elif df['Country name'][i] == 'Guyana':
                    region.append("Latin America and Caribbean")
                elif df['Country name'][i] == 'Guyana':
                    region.append("Latin America and Caribbean")
                elif df['Country name'][i] == 'Bhutan':
                    region.append("South Asia")
                elif df['Country name'][i] == 'Suriname':
                    region.append("Latin America and Caribbean")
                else:
                    region.append(country_continent[df['Country name'][i]])
            df['Regional indicator'] = region
            fig = px.scatter(df, x="Log GDP per capita", y='Life Ladder', animation_frame="year",
                       animation_group="Country name"
                       , color="Regional indicator", hover_name="Country name", size_max=60,
                       color_discrete_sequence=px.colors.qualitative.T10)
            st.plotly_chart(fig)
            st.markdown("Теперь применим R:")
            st.markdown("Первая таблица сделана с использованием gglot2, а также его расширением ggcharts, для ранжирования стран была использована библиотека tidyverse")

            image = Image.open('~/Desktop/R_pic_1.png')
            st.image(image)
            st.markdown("Для второй таблицы сперва получим зависимость Ladder score от ВВП, потом наложим маску, которая разделит все страны на 4 группы относительно России. Воспользуемся расширением ggrepel")
            image = Image.open('~/Desktop/R_pic_2.png')
            st.image(image)
            st.markdown("Посмотрим, что происходит по годам")
            st.markdown("Время для Julia!")
            st.markdown("Для этого с помощью Julia сначала сгруппируем наш датафрейм по годам, потом посчитаем среднее, потом нарисует scatter plot")
            image = Image.open('~/Desktop/Julia_pic_1.png')
            st.image(image)
            st.markdown("Из курса макроэкономики, мы, впрочем, уже могли ожидать получившиеся волны! (ну это похоже на волны...)")

        elif p == "Изучение данных. Медленная таблица":
            st.markdown("Увы, но эта карта ужасно медленная. Так что можно сходить на кухню и заварить чаек, пока она будет грузиться. (спойлер: примерно 2.5 минут). Она грузится в пару этапов, так что не пугайтесь, если что-то изменилось, а карты все нет ")
            st.markdown("Она такая медленная, потому что для 150 стран сначала нужно запросить информацию по координатам и т.п.")
            st.markdown("На карте вы можете увидеть Ladder score 2021 по странам при наведении зума (надо кликнуть по метке). В большом масштабе метки объединяются по областям и принимают соответствующий цвет: зеленый - хорошо, красный - плохо (но с поправкой на кол-во стран в метке) ")
            st.markdown("Сделано с применением folium")
            df_2021 = pd.read_csv('~/Desktop/world-happiness-report-2021.csv')
            df = pd.read_csv('~/Desktop/world-happiness-report.csv')
            geolocator = Nominatim(user_agent="<masked>")
            df_2021 = df_2021.dropna()

            def find_geo(country):
                loc = geolocator.geocode(country)
                return (loc.latitude, loc.longitude)

            my_geo = []
            lon = []
            lat = []
            for i in range(len(df_2021)):
                a = find_geo(df_2021.iloc[i]['Country name'])

                a = str(a).replace("(", "")
                a = a.replace(")", "")
                my_list = str(a).split(", ")
                my_geo.append(a)
                lat.append(my_list[0])
                lon.append(my_list[1])
                # my_geo.append(gpd.GeoDataFrame(df_2021, geometry = gpd.points_from_xy(df_2021['lon'], df_2021['lat'])).set_crs(epsg=4326, inplace=True).to_crs(epsg=3395))
                # print(i)

            df_2021['lat'] = lat
            df_2021['lon'] = lon
            world_map = folium.Map()
            marker_cluster = MarkerCluster().add_to(world_map)
            for i in range(len(df_2021)):
                lat = df_2021.iloc[i]['lat']
                lon = df_2021.iloc[i]['lon']
                text = "Country :" + str(df_2021.iloc[i]['Country name']) + "Ladder score" + str(
                    df_2021.iloc[i]['Ladder score'])
                folium.CircleMarker(location=[lat, lon], radius=3, popup=text, fill=True).add_to(marker_cluster)

            folium_static(world_map)


        elif p == "Планируем переезд":
            st.markdown("Мы изучили данные и стало понятно, что Россия не лучшая страна для прожиания.")
            st.markdown("Давайте планировать переезд.")
            st.markdown("Сначала, с помощью SQL выведем список стран, гдк Ladder score 2021 выше.")
            st.markdown("ВНИМАНИЕ! SQL таблицы кэштрованы, чтобы не было ошибки 'такая таблица уже существует'. Но если что-то сломается, надо зайти в код, долистать до конца, и изменить названия таблиц в SQL. Мне очень жаль, если вам придется это делать... ")
            a = my_sql_1(df_2021, df)
            st.write(a[0])
            st.markdown("Нас обогнала даже Беларусь... Но может это проблема 2021? Посмотрим на топ-10 стран по среднему Ladder score за наш временой период.")
            st.write(a[1])
            st.markdown("Ну... Россия туда не входит. Жаль")
            st.markdown("Выбираем новую страну! Но сперва убедимся, что нам там есть где учиться! С помощью API REST выведем все универы, которые есть в выбранной стране")
            st.markdown("Если вы ткнули в какю-то модную страну, где универов много, просто ткните в предложенный диапозон (по 100 универов в ссылке)")
            r = requests.get('http://universities.hipolabs.com/search').json()
            list_of_countries = []
            for i in range(len(r)):
                list_of_countries.append(r[i]['country'])
            scale_type = st.selectbox("Выберем страну: ", list(set(list_of_countries)))

            def check_uni(country):
                r = requests.get('http://universities.hipolabs.com/search', params={'country': country}).json()
                print(len(r))
                if r:
                    my_list = []
                    for i in range(len(r)):
                        # counter +=1
                        my_list.append(r[i]['name'])

                    return my_list
                else:
                    return 'No uni in this country'
            st.write(check_uni(scale_type))

        elif p == "Немножко ML":
            st.markdown("Теперь что-то интересное! Построим модель-предсказание Ladder score по другим параметрам с помощью регрессии!")
            st.markdown("Но сначала изучим данные. Есть ли у нас корреляции?")



            cols = ["Ladder score", "Logged GDP per capita", "Social support", "Healthy life expectancy",
                    "Freedom to make life choices", "Generosity", "Perceptions of corruption"]
            df = df.rename( columns ={'Life Ladder': 'Ladder score', 'Log GDP per capita': 'Logged GDP per capita',
                                    'Healthy life expectancy at birth': 'Healthy life expectancy'})
            data = df[cols].dropna()
            test = df_2021[cols].dropna()

            fig = sns.pairplot(data)
            st.pyplot(fig)
            cols_1 = ["year", "Ladder score", "Logged GDP per capita", "Social support", "Healthy life expectancy",
                    "Freedom to make life choices", "Generosity", "Perceptions of corruption", "Positive affect",
                    "Negative affect"]

            df_new = df[cols_1]
            st.markdown("Построим таблицу корреляций")
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111)
            sns.heatmap(df_new.corr(), annot=True, vmin=-1, vmax=1)
            ax.set_xticklabels(df_new.columns)

            ax.set_title("Correlation", fontsize=12)
            st.pyplot(fig)
            st.markdown("Но мы же не зря учили теор.вер и статистику! Давайте, используя математические возможности Python, откинем неинтересные клетки (где p-value > 10% и все верхние)")
            def corr_sig(df):
                p_matrix = np.zeros(shape=(df.shape[1], df.shape[1]))
                for i in df.columns:
                    for j in df.drop(i, axis=1).columns:
                        df_new = df[[i, j]].dropna()
                        _ , p = stats.pearsonr(df_new[i], df_new[j])
                        p_matrix[df.columns.to_list().index(i), df.columns.to_list().index(j)] = p
                return p_matrix

            dfg = df[cols_1]

            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111)
            p_values = corr_sig(dfg).round(10)
            mask = np.invert(np.tril(p_values < 0.1))
            sns.heatmap(df.corr(), annot=True, mask=mask, vmin=-1, vmax=1)
            ax.set_xticklabels(dfg.columns)

            ax.set_title("Сorrelation where p-value < 10%", fontsize=12)
            st.pyplot(fig)

            st.markdown("Теперь результаты нашей модели: синяя линия - действительные значения в 2021, красная - предсказанные. Модель не идеальна, но тренд сохранен.")
            st.markdown("Для обучения модели разделим данные на тест и тренинг, обработем их, используя продвинутые возможности pandas")
            data_predictors = data.drop("Ladder score", axis=1)
            data_labels = data["Ladder score"].copy()
            data_train = Pipeline([('std_scaler', StandardScaler())]).fit_transform(data_predictors)
            regression = LinearRegression()
            regression.fit(data_train, data_labels)
            tested_data_1 = test.drop("Ladder score", axis=1)
            tested_data_labels = test["Ladder score"]
            tested_data = Pipeline([('std_scaler', StandardScaler())]).fit_transform(tested_data_1)
            predictions = regression.predict(tested_data)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_2021["Country name"], y=tested_data_labels, name='real labels'))
            fig.add_trace(go.Scatter(x=df_2021["Country name"], y=predictions, name='predicted labels'))
            fig.update_layout(legend_orientation="h",
                              legend=dict(x=.5, xanchor="center"),
                              title="The predictions of our model and the real labels",
                              xaxis_title="Country",
                              yaxis_title="Ladder score",
                              margin=dict(l=0, r=0, t=70, b=70))
            st.plotly_chart(fig)
        elif p == "Победитель конкурса 'Лучшая страна'" :
            st.markdown("Как вы поняли, лучшая страна - Финляндия! Поэтому ей достается честь продемонстрировать веб-скреппинг")
            st.markdown("Парсить будем с Википедии!")

            r = requests.get('https://ru.wikipedia.org/wiki/Финляндия')
            text = r.text

            r = BeautifulSoup(text, features="html.parser")

            spans = r.find_all("span", {'class': "nowrap"})
            population = spans[4].string.split(' чел.')[0]
            terr = spans[1].string

            st.markdown("Ну, для начала, найдем население. В Финляндии живут " + str(population) + " человек")
            st.markdown("Теперь территорию! Территория Финляндии " + str(terr) + " км^2" )
            st.markdown("Ну и, напоследок, поанализируем немножко Финляндию.")
            year = []
            ladder = []
            gdp = []
            socialsup = []
            healt = []
            for i in range(len(df)):
                if df["Country name"][i] == "Finland":
                    year.append(df["year"][i])
                    ladder.append(df["Life Ladder"][i])
                    gdp.append(df["Log GDP per capita"][i])
                    socialsup.append(df["Social support"][i])
                    healt.append(df["Healthy life expectancy at birth"][i] / 7)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=year, y=ladder, name='Ladder score'))
            fig.add_trace(go.Scatter(x=year, y=gdp, name='gdp'))
            fig.add_trace(go.Scatter(x=year, y=socialsup, name='Social support'))
            fig.add_trace(
                go.Scatter(x=year, y=healt, name='Healthy life expectancy нормализованая к шкале остальных покзателей'))
            fig.update_layout(legend_orientation="h",
                              legend=dict(x=.5, xanchor="center"),
                              title="Изменения показателей Финляндии со времнем",
                              xaxis_title="Year",
                              # yaxis_title="Ladder score",
                              margin=dict(l=0, r=0, t=70, b=70))
            st.plotly_chart(fig)


    @st.cache
    def my_sql_1(df_2021, df):
        conn = sqlite3.connect("database.sqlite")
        conn.commit()

        df_2021.to_sql("df_2021_new8", conn)

        def sql(request):
            return pd.read_sql_query(request, conn)

        table1 = sql("""
                    SELECT "Country name", "Ladder score" FROM df_2021_new8
                    WHERE "Ladder score" > 5.49
                    """)

        df.to_sql("df_new8", conn)

        table2 = sql("""
                            SELECT "Country name", AVG("Life Ladder"), MAX("Life Ladder") FROM df_new8
                            GROUP BY "Country name"
                            ORDER BY AVG("Life Ladder") DESC
                            LIMIT 10
                            """)
        return [table1, table2]

    if __name__ == "__main__":
        main()

